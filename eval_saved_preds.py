import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import json
import argparse

def video_to_array(video_path, radius=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to grayscale and keep only one channel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if radius > 1:
            from scipy.ndimage import maximum_filter
            frame = maximum_filter(frame, size=radius)
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    # print(f"Min value: {frames.min()}, Max value: {frames.max()}")
    # Reshape to (t,c,h,w) format
    if len(frames) > 0:
        if frames.ndim == 1:
            return None
        return frames[:, None, :, :]
    else:
        return None

def mae(*args):
    return np.mean(np.abs(args[0] - args[1]))

def mse(*args):
    return np.mean((args[0] - args[1])**2)

def tfpn(y_pred, y, threshold, radius=1):
    '''
    Convert numpy arrays to binary and compute confusion matrix values
    '''
    if radius > 1:
        from scipy.ndimage import maximum_filter
        y = maximum_filter(y, size=radius)
        y_pred = maximum_filter(y_pred, size=radius)
    
    y = np.where(y >= threshold, 1, 0)
    y_pred = np.where(y_pred >= threshold, 1, 0)
    
    tp = np.sum((y_pred == 1) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    return tp, tn, fp, fn

def csi(tp, tn, fp, fn):
    '''Critical Success Index. The larger the better.'''
    if (tp + fn + fp) < 1e-7:
        return 0.
    return tp / (tp + fn + fp)

def csi_4(tp, tn, fp, fn):
    return csi(tp, tn, fp, fn)

def csi_16(tp, tn, fp, fn):
    return csi(tp, tn, fp, fn)

def far(tp, tn, fp, fn):
    '''False Alarm Rate. The smaller the better.'''
    if (tp + fp) < 1e-7:
        return 0.   
    return fp / (tp + fp)

def pod(tp, tn, fp, fn):
    '''Probability of Detection (ML: Recall). The larger the better.'''
    if (tp + fn) < 1e-7:
        return 0.    
    return tp / (tp + fn)

def ssim(y_pred, y):
    from skimage.metrics import structural_similarity
    if y.ndim == 4:
        b, t, h, w = y.shape
        y = y.reshape(b*t, h, w)
        y_pred = y_pred.reshape(b*t, h, w)
    
    # Ensure values are in [0,1]
    y = np.clip(y, 0, 1)
    y_pred = np.clip(y_pred, 0, 1)
    
    score = 0
    for i in range(len(y)):
        score += structural_similarity(y[i], y_pred[i], data_range=1.0)
    return score / len(y)

def psnr(y_pred, y):
    from skimage.metrics import peak_signal_noise_ratio
    if y.ndim == 4:
        b, t, h, w = y.shape
        y = y.reshape(b*t, h, w)
        y_pred = y_pred.reshape(b*t, h, w)
    
    score = 0
    for i in range(len(y)):
        score += peak_signal_noise_ratio(y[i], y_pred[i], data_range=1.0)
    return score / len(y)

def inception_score(y_pred, y):
    raise NotImplementedError()

def fid(y_pred, y):
    raise NotImplementedError()

def lpips(y_pred, y, net='vgg'):
    '''
    Note: LPIPS still requires torch tensors internally, so we convert numpy arrays
    '''
    import torch
    import lpips as lp
    
    # Convert to torch tensors
    y = torch.from_numpy(y).float()
    y_pred = torch.from_numpy(y_pred).float()
    
    # Reshape if needed
    if y.ndim == 4:
        b, t, h, w = y.shape
        y = y.reshape(-1, 1, h, w)
        y_pred = y_pred.reshape(-1, 1, h, w)
    
    # Scale to [-1,1]
    y = (2 * y - 1)
    y_pred = (2 * y_pred - 1)
    
    # Use LPIPS
    global GLOBAL_LPIPS_OBJ
    if GLOBAL_LPIPS_OBJ is None:
        GLOBAL_LPIPS_OBJ = lp.LPIPS(net=net)
        if torch.cuda.is_available():
            GLOBAL_LPIPS_OBJ = GLOBAL_LPIPS_OBJ.cuda()
            y = y.cuda()
            y_pred = y_pred.cuda()
    
    with torch.no_grad():
        score = GLOBAL_LPIPS_OBJ(y_pred, y).mean().cpu().numpy()
    return score

class MetricListEvaluator():    
    '''
    To evaluate a list of metrics. Supported metrics:
    - CSI (Eg. `csi-84`)
    - POD (Eg. `pod-84`)
    - FAR (Eg. `far-84`)
    - MAE
    - MSE
    - SSIM
    - PSNR    
    '''
    def __init__(self, metric_list, out_len):        
        self.metric_holder = {}
        self.batch_count = 0
        self.out_len = out_len
        for metric_name in metric_list:
            threshold = ''
            if '-' in metric_name:
                metric_name, threshold = metric_name.split('-')
            # initialize metrics
            key_name = metric_name + (f'-{threshold}' if len(threshold) > 0 else '')
            threshold = float(threshold) / 255 if threshold.isdigit() else threshold
            self.metric_holder[key_name] = self.init_metric(metric_name, threshold=threshold)
    
    def init_metric(self, metric_name, **kwarg):
        '''
        return a tuple of three items in order:
        - the function to call during eval
        - the value(s) to keep track of
        - a dict of any additional item to pass into the function
        '''        
        if metric_name in ['csi', 'pod', 'far']:
            # use tfpn instead
            return [tfpn, np.zeros((self.out_len, 4), dtype=np.float32), {'threshold': kwarg['threshold']}] # tp, 
        elif metric_name == 'csi_4':
            # tfpn with radius (pooling)
            return [tfpn, np.zeros((self.out_len, 4), dtype=np.float32), {'threshold': kwarg['threshold'], 'radius': 4}]
        elif metric_name == 'csi_16':
            return [tfpn, np.zeros((self.out_len, 4), dtype=np.float32), {'threshold': kwarg['threshold'], 'radius': 16}]
        else:
            # directly convert the string name into function call
            return [eval(metric_name), 0, {}]

    def eval(self, y_pred, y):
        self.batch_count += 1
        for metric_name, metric in self.metric_holder.items():
            if '-' in metric_name: 
                for i in range(self.out_len):
                    temp_y_pred = y_pred[i, :, :, :]
                    temp_y = y[i, :, :, :]
                    temp = metric[0](temp_y_pred, temp_y, **metric[-1])   
                    if isinstance(temp, list):
                        temp = np.array(temp)
                    metric[1][i] += temp
            else:
                temp = metric[0](y_pred, y, **metric[-1])      
                if temp is list:
                    temp = np.array(temp)
                metric[1] += temp                

    def get_results(self):
        print(self.batch_count)
        output_holder = {}
        for key, metric in self.metric_holder.items():
            val = metric[1]
            # special handle of tfpn => compute the final score now
            if metric[0] is tfpn:
                metric_name, threshold = key.split('-')
                val = val[:,0]
                for i in range(self.out_len):
                    val[i] = eval(metric_name)(*list(metric[1][i]))
            else:
                val /= self.batch_count if self.batch_count > 0 else 1
            output_holder[key] = val
        return output_holder

def evaluate_predictions(dir_path, radius=1):
    metric_list = ['mae', 'mse', 'ssim', 'psnr', 'csi-84', 'csi-117', 'csi-140', 'csi-158', 'csi-185', 'pod-84', 'far-84']
    evaluator = MetricListEvaluator(metric_list, out_len=20)

    # Get all files in the directory
    file_pattern = os.path.join(dir_path, '*_pred.mp4')
    pred_files = glob.glob(file_pattern)

    # Process each set of files
    for pred_file in tqdm(pred_files, desc='Processing videos'):
        # Get corresponding input and output files
        base_name = pred_file[:-9] # Remove '_pred.mp4'
        input_file = base_name + '_in.mp4'
        output_file = base_name + '_out.mp4'
        
        # Load the videos
        pred_frames = video_to_array(pred_file, radius=radius)
        output_frames = video_to_array(output_file, radius=radius)
        
        if pred_frames is not None and output_frames is not None:
            # Remove last frame and normalize
            pred_frames = pred_frames[:-1] / 255
            output_frames = output_frames[:-1] / 255
            
            # Evaluate
            evaluator.eval(pred_frames, output_frames)
        else: 
            print(f'Skipped {pred_file}')

    final_results = evaluator.get_results()
    for k, v in final_results.items():
        print(f'{k}: {v}')

    dir_name = os.path.basename(dir_path)
    json_path = os.path.join(dir_path, f'{dir_name}_radius{radius}_results.json')

    # Convert numpy arrays to lists for JSON serialization
    final_results_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in final_results.items()}

    with open(json_path, 'w') as f:
        json.dump(final_results_serializable, f, indent=4)
    print(f'Results saved to {json_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate video predictions')
    parser.add_argument('dir_path', type=str, help='Directory containing prediction videos')
    parser.add_argument('--radius', type=int, default=1, help='Radius for video processing')
    args = parser.parse_args()
    
    evaluate_predictions(args.dir_path, args.radius)

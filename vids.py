from itertools import islice
from typing import Optional


import cv2
import numpy as np
from PIL import Image
import dask.array as da
# import av


class Video:
    
    def __init__(self, filename: str, thumbnail_height: int = 200):
        # self.cont = av.open(filename)
        self.cap = cv2.VideoCapture(filename)
        self.thumbnail_height = thumbnail_height
        
    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def nchans(self) -> int:
        return 3
    
    @property
    def nframes(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __len__(self):
        return self.nframes
        
    def _repr_png_(self):
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = self.thumbnail_height
        h = int(self.height / (self.width / w))
        full_img = Image.new('RGB', (w * 5, h))
        for rep, idx in enumerate(np.linspace(0, self.nframes - 1, 5, dtype=int)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read()
            img = Image.fromarray(frame, 'RGB')
            img.thumbnail((w, h))
            full_img.paste(img, (w * rep, 0))
        
        return full_img._repr_png_()
    
    def __getitem__(self, idx: int) -> np.ndarray:
        # allow slicing 
        if type(idx) is slice:
            return islice(self, idx.start, idx.stop, idx.step) 
        self.seek(idx)
        frame = self.read()
        return frame
    
    def __iter__(self):
        self.seek(0)
        return self
        
    def __next__(self):
        try:
            return self.read()
        except EOFError:
            raise StopIteration  
    
    def seek(self, idx: int) -> None:
        nframes = self.nframes
        
        # Alllow negative indices
        if -nframes <= idx < 0:
            idx = nframes + idx  
        if not 0 <= idx < nframes:
            raise IndexError(f"requested frame {idx}, video has only {nframes} frames.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        
    def read(self):
        res, frame = self.cap.read()
        if not res:
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                raise EOFError("reached end of file.")
            else:
                raise IOError("unknown error")
        return frame       
        
        
        
def preview(frame: np.ndarray, width=400, height=400):
    img = Image.fromarray(frame, 'RGB')
    img.thumbnail((height, width))
    return img



from time import perf_counter_ns, sleep

class Timer:
    def __init__(self):
        self.measured = None
        self.measurements = []
        self.zero_offset = 0.
        self.reset()
        
    def __enter__(self):
        self.reset()
        return self
        
    def __exit__(self, type, value, traceback):
        self.measured = self.read()
        
    def calibrate(self, reps=7, wait=.02) -> None:
        timer = Timer()
        for _ in range(reps):
            sleep(wait)
            timer.write()
            timer.reset()
        self.zero_offset = sum(timer.measurements) / len(timer.measurements) - wait
        
    def reset(self) -> None:
        self.__time = perf_counter_ns()
        
    def read(self) -> float:
        return (perf_counter_ns() - self.__time) * 1e-9 - self.zero_offset
        
    def write(self) -> None:
        self.measurements.append(self.read())
        
        


        
        
        

            
        
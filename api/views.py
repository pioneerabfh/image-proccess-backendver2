# api/views.py
import io, base64, json, math
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import numpy as np


def pil_to_np(img):         
    return np.asarray(img).astype(np.float32) / 255.0

def np_to_pil(arr):          
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gray(img):
    g = img[..., 0]*0.299 + img[...,1]*0.587 + img[...,2]*0.114
    return np.stack([g, g, g], axis=-1)

def binary(img, thresh=0.5):
    g = gray(img)[...,0]
    bw = (g > thresh).astype(np.float32)
    return np.stack([bw, bw, bw], axis=-1)

def rotate_np(img, angle):
    
    pil = np_to_pil(img)
    rotated = pil.rotate(angle, expand=True)
    return pil_to_np(rotated)

def crop_np(arr, startX, startY, width, height):
   
    H, W = arr.shape[:2]          

    
    try:
        x, y, w, h = map(int, [startX, startY, width, height])
    except ValueError:
        raise ValueError("Kırpma parametreleri sayısal olmalı")

    if w <= 0 or h <= 0:
        raise ValueError("Genişlik ve yükseklik > 0 olmalı")

    
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = min(w, W - x)             
    h = min(h, H - y)

    if w == 0 or h == 0:
        raise ValueError("Kırpma koordinatları resmin dışında kaldı")

    return arr[y : y + h, x : x + w, :]

def zoom(img, k, x0, y0):
    h, w = img.shape[:2]
    dst_h, dst_w = int(h*k), int(w*k)
    pil = np_to_pil(img).resize((dst_w, dst_h), Image.BICUBIC)
    arr = pil_to_np(pil)
    
    return arr[y0:y0+h, x0:x0+w, :]

def brightness(img, val):
    return np.clip(img + val, 0, 1)


def conv2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h,pad_h), (pad_w,pad_w), (0,0)), mode='reflect')
    out = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y+kh, x:x+kw, :]
            out[y, x] = (region * kernel[...,None]).sum(axis=(0,1))
    return out

def gaussian_kernel(ksize, sigma):
    ax = np.arange(-ksize//2 + 1., ksize//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2.*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)

def sobel(img):
    gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    gy = gx.T
    g_img = gray(img)[...,0]
    grad_x = conv2d(g_img[...,None], gx)[:,:,0]
    grad_y = conv2d(g_img[...,None], gy)[:,:,0]
    mag = np.sqrt(grad_x**2 + grad_y**2)
    mag = mag / mag.max()
    return np.stack([mag, mag, mag], axis=-1)

def salt_pepper(img, salt, pepper):
    out = img.copy()
    h, w = out.shape[:2]
    num_salt = int(salt * h * w)
    num_pepper = int(pepper * h * w)
    ys = np.random.randint(0, h, num_salt)
    xs = np.random.randint(0, w, num_salt)
    out[ys, xs] = 1.0
    ys = np.random.randint(0, h, num_pepper)
    xs = np.random.randint(0, w, num_pepper)
    out[ys, xs] = 0.0
    return out

def mean_filter(img, k):
    kernel = np.ones((k, k), dtype=np.float32) / (k*k)
    return conv2d(img, kernel)

def median_filter(img, k):
    pad = k//2
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    out = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y+k, x:x+k, :]
            out[y,x] = np.median(region, axis=(0,1))
    return out

def adaptive_threshold(img, block, c):
    g = gray(img)[...,0]
    pad = block//2
    padded = np.pad(g, pad, mode='reflect')
    out = np.zeros_like(g)
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            region = padded[y:y+block, x:x+block]
            thresh = region.mean() - c/255.0
            out[y,x] = 1.0 if g[y,x] > thresh else 0.0
    return np.stack([out,out,out], axis=-1)


def dilate(img, k):
    pad = k//2
    g = gray(img)[...,0]
    padded = np.pad(g, pad, mode='constant', constant_values=0)
    out = np.zeros_like(g)
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            out[y,x] = np.max(padded[y:y+k, x:x+k])
    return np.stack([out,out,out], axis=-1)

def erode(img, k):
    pad = k//2
    g = gray(img)[...,0]
    padded = np.pad(g, pad, mode='constant', constant_values=1)
    out = np.zeros_like(g)
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            out[y,x] = np.min(padded[y:y+k, x:x+k])
    return np.stack([out,out,out], axis=-1)

def open_morph(img, k):  
    return dilate(erode(img, k), k)

def close_morph(img, k):  
    return erode(dilate(img, k), k)


@csrf_exempt
def image_proccess(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        img_b64 = data['imageBase64'].split(';base64,')[1]
        img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        arr = pil_to_np(img)

        for op in data.get('operations', []):
            name = op['name']
            p = op.get('params', {})

            if name == 'crop':
                p = op['params']
                arr = crop_np(arr, p['startX'], p['startY'], p['width'], p['height'])
            elif name == 'rotate':
                arr = rotate_np(arr, p['angle'])
            elif name == 'gray':
                arr = gray(arr)
            elif name == 'binary':
                arr = binary(arr, p.get('thresh', 0.5))
            elif name == 'zoom':
                arr = zoom(arr, p['k'], p['startX'], p['startY'])
            elif name == 'brightness':
                arr = brightness(arr, p['val'])
            elif name == 'gauss':
                kernel = gaussian_kernel(p['ksize'], p['sigma'])
                arr = conv2d(arr, kernel)
            elif name == 'sobel':
                arr = sobel(arr)
            elif name == 'salt_pepper':
                arr = salt_pepper(arr, p['salt'], p['pepper'])
            elif name == 'mean_filter':
                arr = mean_filter(arr, p['ksize'])
            elif name == 'median_filter':
                arr = median_filter(arr, p['ksize'])
            elif name == 'adaptive_threshold':
                arr = adaptive_threshold(arr, p['block'], p['c'])
            elif name == 'dilate':
                arr = dilate(arr, p['ksize'])
            elif name == 'erode':
                arr = erode(arr, p['ksize'])
            elif name == 'open':
                arr = open_morph(arr, p['ksize'])
            elif name == 'close':
                arr = close_morph(arr, p['ksize'])
            elif name == 'add_image':
                
                arr2_b64 = p['image2'].split(';base64,')[1]
                img2 = pil_to_np(Image.open(io.BytesIO(base64.b64decode(arr2_b64))).convert("RGB"))
                if p['op'] == 'add':
                    arr = np.clip(arr + img2, 0, 1)
                elif p['op'] == 'multiply':
                    arr = np.clip(arr * img2, 0, 1)

        
        out_pil = np_to_pil(arr)
        buff = io.BytesIO()
        out_pil.save(buff, format='PNG')
        encoded = base64.b64encode(buff.getvalue()).decode('utf-8')
        return JsonResponse({"result": "data:image/png;base64," + encoded})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

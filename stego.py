import cv2
import numpy as np
from utils.preprocessing import CannyP
from utils.preprocessing import CropLayer
import sys
l=[]
def edge(path):
    # get image path
    if(len(sys.argv) > 1):
        src_path = sys.argv[1]
    else:
        src_path = path
    # read image
    img = cv2.imread(src_path, 1) 
    if(img is None):
        print("Image not read properly")
        sys.exit(0)
    # initialize preprocessing object
    obj = CannyP(img)
    width = 500
    height = 500
    # remove noise
    img = obj.noise_removal(filterSize=(5, 5))
    prototxt = "deploy.prototxt"
    caffemodel = "hed_pretrained_bsds.caffemodel"
    global l
    if path in l:    cv2.dnn_registerLayer('Crop', CropLayer)
    l.append(path)
    net = cv2.dnn.readNet(prototxt, caffemodel)
    inp = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(width, height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv2.resize(out, (img.shape[1], img.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    # visualize
    #cv.imshow("HED", out)
    #cv.imshow("original", img)
    #k = cv2.waitKey(0) & 0xFF
    #if(k == 27):
      #  cv2.destroyAllWindows()
    arr = np.array(out)
    '''shape = arr.shape
    flat_arr = arr.ravel()
    l=[]
    print(len(flat_arr))
    for i in range(0,len(flat_arr)):
        if(flat_arr[i]>50): l.append(i)'''
    l=[]
    #print(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if(arr[i][j]>50): l.append([i,j])
    
    return l
def to_bin(data):
    """Convert `data` to binary format as string"""
    if isinstance(data, str):
        return ''.join([ format(ord(i), "08b") for i in data ])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [ format(i, "08b") for i in data ]
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, "08b")
    else:
        raise TypeError("Type not supported.")
def encode(image_name, secret_data,output_image):
    # read the image
    image = cv2.imread(image_name)
    print("data for encoding: "+secret_data,end="")
    # maximum bytes to encode
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    print("[*] Maximum bytes to encode:", n_bytes)
    if len(secret_data) > n_bytes:
        raise ValueError("[!] Insufficient bytes, need bigger image or less data.")
    print("[*] Encoding data...")
    # add stopping criteria
    secret_data += "====="
    data_index = 0
    # convert data to binary
    binary_secret_data = to_bin(secret_data)
    # size of data to hide
    data_len = len(binary_secret_data)
    l=edge(image_name)
    for i in l:
            p,q=i[0],i[1]
            pixel=image[p][q]
            
            # convert RGB values to binary format
            r, g, b = to_bin(pixel)
            # modify the least significant bit only if there is still data to store
            if data_index < data_len:
                # least significant red pixel bit
                pixel[0] = int(r[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            if data_index < data_len:
                # least significant green pixel bit
                pixel[1] = int(g[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            if data_index < data_len:
                # least significant blue pixel bit
                pixel[2] = int(b[:-1] + binary_secret_data[data_index], 2)
                data_index += 1
            # if data is encoded, just break out of the loop
            if data_index >= data_len:
                break
    cv2.imwrite(output_image, image)
    print("*Encoding Successful*")
def decode(image_name):
    print("[+] Decoding...")
    # read the image
    image = cv2.imread(image_name)
    binary_data = ""
    
    l=edge(image_name)
    for i in l:
            p,q=i[0],i[1]
            pixel=image[p][q]
            
            r, g, b = to_bin(pixel)
            binary_data += r[-1]
            binary_data += g[-1]
            binary_data += b[-1]
    # split by 8-bits
    all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "=====":
            break
    print("*Decoding Successful*")
    return decoded_data[:-5]






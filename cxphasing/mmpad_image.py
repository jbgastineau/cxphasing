import Image
import scipy as sp
import pdb

def tif_to_numpy(filename):
    return sp.misc.fromimage(Image.open(filename))

def masks():
    
    mask = sp.ones((7,266,396))
    mask[:,134:,0:1]=0.
    mask[:,134:,129:130]=0.
    mask[:,:,262:266] = 0. 
    mask[:,:,130:134] = 0.
    mask[:,-2:,:] = 0.
    mask[:,130:136,:] = 0.
    mask[:,:2,:] = 0.
    mask[:,:134,:2]=0.
    mask[:,:,-2:]=0.
    
    mask[1,134:,:]=0.
    mask[1,:,132:]=0.

    mask[2,:,1:134]=0.
    mask[2,:,265:]=0.
    mask[2,134:,:]=0.
    
    mask[3,:,:263]=0.
    mask[3,134:,:]=0.
    
    mask[4,:,134:]=0.
    mask[4,:137,:]=0.
    
    mask[5,:,:134]=0.
    mask[5,:,264:]=0.
    mask[5,:137,:]=0.
    
    mask[6,:,:263]=0.
    mask[6,:137,:]=0.
    
    return mask

def remove_aliens(arr):
    aliens = [(93,393), (92,393), (94,393),(91,393)]
    for alien in aliens:
        arr[alien]=0.
    arr=sp.where(arr<-20,-20,arr)
    return arr

def pedestal_correct(arr,mask=None):
    if mask==None: mask=masks()
    
    arrc = sp.where(arr<5, arr, 0.)
    corrected_image = sp.zeros(arr.shape)
    for i in range(6):
        masked_data = arrc*mask[i+1,:,:]
        
        corrected_image += mask[i+1,:,:]*(arr-(sp.sum(masked_data)/sp.sum(mask[i+1,:,:])))
        
    return corrected_image
    
def threshhold(arr,val=None):
    if val == None: val=5.
    return sp.where(arr>val,arr,val)-val

def open_mmpad_tif(filename):
    img = tif_to_numpy(filename)
    img = remove_aliens(img)
    img = pedestal_correct(img)
    return threshhold(img)

    
    
    
    
    
    
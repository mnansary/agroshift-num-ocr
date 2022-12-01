# Recognition: 

## **only covers bengali and english numbers with additional extenders like**:
* ```-```: 01xxx-xxxxxx : mobile numbers
* ```/```: xx/-: amount 
* ```=```: xx/=: amount (differentiator with the previous one)

**vocab**

```python
["blank","-","/","=","০","১","২","৩","৪","৫","৬","৭","৮","৯","0","1","2","3","4","5","6","7","8","9","sep","pad"]
```  


## **Three possible solutions**:

### ***custom SVT***: 
* advantages:
    * faster inference: ~10 images in 1 sec
    * onnx-convertable: optimized for moduling
    * works with direct handwritten images
    * end-to-end
* disadvantages:
    * requires a lot of ```real``` data: ~500k samples of strings
    * batchsize-64 

### ***Robust Scanner***: 
* advantages:
    * works with direct handwritten images
    * requires less ```real``` data: ~150k samples of strings
    * batchsize-256
    * end-to-end
* disadvantages:
    * slower inference: ~3 images in 1 sec
    * not onnx-convertable: not optimized for moduling
    
### ***Modifier OCR***: 
* advantages:
    * requires less ```real``` data: ~50k samples of strings
    * slower inference: ~7 images in 1 sec
    * onnx-convertable: optimized for moduling

* disadvantages:
    * batchsize-128
    * does not work with direct handwritten images
    * not end-to-end (needs an intermediary modifier model)

## **DataSet**:

* synthetic samples:

![](https://github.com/mnansary/agroshift-num-ocr/blob/main/Docs/srcs/synth.png?raw=true)
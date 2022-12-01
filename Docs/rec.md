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

### ***custom SVT***: Stage:Execution
* advantages:
    * faster inference: ~10 images in 1 sec
    * onnx-convertable: optimized for moduling
    * works with direct handwritten images
    * end-to-end
* disadvantages:
    * requires a lot of ```real``` data: ~500k samples of strings
    * batchsize-64 

- [X] Dataset: https://www.kaggle.com/datasets/ocrteamriad/argo-hwnums-ds
- [X] Weights:https://www.kaggle.com/datasets/ocrteamriad/argo-weights-cvt 
- [X] Train Code: https://www.kaggle.com/code/ocrteamriad/argo-train-cvt
- [X] Onnx Code: https://www.kaggle.com/code/nazmuddhohaansary/agro-onnx-converter-rec


### ***Robust Scanner***: Stage:Conditional Improvement
* advantages:
    * works with direct handwritten images
    * requires less ```real``` data: ~150k samples of strings
    * batchsize-256
    * end-to-end
* disadvantages:
    * slower inference: ~3 images in 1 sec
    * not onnx-convertable: not optimized for moduling
    
### ***Modifier OCR***: Stage:FailSafe
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

* Data Distribution:

|Feature|Description|ROI|
|:--:|:---:|:---:|
|NUM_MOB=1M| <ul><li>Synthetic Mobile Number Images</li><li>Bangla:English ratio 30:70</li></ul> |Worker_phone|
|NUM_AMM=200K| <ul><li>Synthetic Amount Images</li><li>Bangla:English ratio 70:30</li><li> fixed length 1,2 (randomchoice)</li>|Sup_Id<br />28_chal_qty_bosta<br />29_chal_qty_bosta<br />2L_oil_pcs<br />5L_oil_pcs<br />Lintel_kg<br />Flower_kg<br />potato_kg<br />onion_kg<br />salt_kg<br />sugar_kg<br />|
|NUM_IDX=600K| <ul><li>Synthetic ID Images</li><li>Bangla:English ratio 35:65</li><li> fixed length 4,5,6,7 (randomchoice)</li>|Worker_ID<br />Sup_Id|
|NUM_TOT=200k| <ul><li>Synthetic Total Value Images</li><li>Bangla:English ratio 70:30</li><li> fixed length 3,4 (randomchoice)</li>|TOV_worker|

* Data Specification:

|Feature|Description|Comment|
|:--:|:---:|:---:|
|resizeToHeight|70%<br /> ![](https://github.com/mnansary/agroshift-num-ocr/blob/main/Docs/srcs/6.png?raw=true)|ad:Forced alignment <br /> dis: ratio of 0 to other numbers mismatch|
|padToHeight|30%<br /> ![](https://github.com/mnansary/agroshift-num-ocr/blob/main/Docs/srcs/6.png?raw=true)|ad:natural alignment <br /> dis:high distortion|

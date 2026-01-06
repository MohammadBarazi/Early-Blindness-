# APTOS 2019 Blindness Detection — EfficientNetB3 (Regression) + 5-Fold Stratified CV + QWK + Optimized Thresholds

هذا المشروع يطبّق حلًا لمسابقة Kaggle **APTOS 2019 Blindness Detection** لتقدير شدة اعتلال الشبكية السكري (Diabetic Retinopathy) من صور قاع العين (Fundus).  
الحل مبني على **Transfer Learning** باستخدام **EfficientNetB3**، مع **Cross-Validation**، وقياس أداء فعلي بواسطة **Quadratic Weighted Kappa (QWK)**، ثم **تحسين حدود التحويل (Thresholds)** على تنبؤات الـ OOF لتعظيم QWK.

---

## 1) الهدف
- **المدخل**: صورة Fundus (`.png`)
- **المخرج**: درجة `diagnosis` ضمن {0,1,2,3,4}
- **التقييم**: QWK (Quadratic Weighted Kappa)

> ملاحظة تصميم: الكود يتعامل مع المشكلة كـ **Regression** (خرج واحد `Dense(1)`) ثم يحوّل الناتج إلى فئات 0..4 عبر:
> - تقريب (rounding) أثناء مراقبة الأداء داخل التدريب
> - و/أو Threshold Optimization بعد التدريب للحصول على أفضل QWK على OOF

---

## 2) البيانات (Dataset)
يتوقع الكود هيكلة Kaggle القياسية:

- `../input/aptos2019-blindness-detection/train.csv`
- `../input/aptos2019-blindness-detection/test.csv`
- `../input/aptos2019-blindness-detection/train_images/*.png`
- `../input/aptos2019-blindness-detection/test_images/*.png`

أعمدة البيانات:
- `train.csv`:  
  - `id_code`: معرف الصورة  
  - `diagnosis`: الدرجة (0..4)
- `test.csv`:  
  - `id_code`
- صيغة التسليم مطابقة لـ `sample_submission.csv` (عمودان: `id_code`, `diagnosis`)

---

## 3) الإعدادات (Configuration) كما في الكود
- `SEED = 2019`
- `IMG_SIZE = 300`
- `BATCH_SIZE = 8`
- `N_SPLITS = 5` (Stratified K-Fold)
- Warm-up:
  - `EPOCHS_WARMUP = 2`
  - Optimizer: `Adam(1e-3)`
  - Loss: `MSE`
- Fine-tuning:
  - `EPOCHS_FINE = 5`
  - Optimizer: `Adam(5e-5)`
  - Loss: `Huber(delta=1.0)`
  - `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6)`

---

## 4) Pipeline عام (كيف يشتغل المشروع)

### 4.1 تحميل المسارات والوسوم
- يتم بناء مسارات الصور من `id_code`:
  - `train_images/{id_code}.png`
  - `test_images/{id_code}.png`
- Targets:
  - `labels_reg`: float32 لاستخدامها في regression
  - `labels_strat`: int لاستخدامها في stratification

### 4.2 Preprocessing (Minimal + عملي)
**الدوال الأساسية:**
- `crop_image_from_gray(img, tol=7)`  
  تقص الخلفية الداكنة/السوداء عبر threshold على الصورة الرمادية (gray mask)، لتقليل الحواف السوداء الشائعة في صور fundus.
- `minimal_preprocess(path, img_size=IMG_SIZE)`  
  1) قراءة الصورة بـ OpenCV  
  2) BGR → RGB  
  3) قص الخلفية الداكنة (`crop_image_from_gray`)  
  4) Resize إلى 300×300  
  5) Clip إلى [0..255] و تحويل إلى `float32`

> لا يوجد “Ben Graham enhancement” هنا؛ المعالجة Minimal عمدًا.

### 4.3 tf.data input pipeline
- `tf.numpy_function` لتشغيل OpenCV preprocessing داخل `tf.data`
- `shuffle` فقط أثناء التدريب
- `batch(BATCH_SIZE)` ثم `prefetch(AUTOTUNE)`

**مخرجات الـ Dataset:**
- تدريب/تحقق: `(image, y, sample_weight)`
- اختبار: `image` فقط

### 4.4 Augmentation داخل النموذج
داخل `build_model`:
- `RandomFlip("horizontal")`
- `RandomRotation(0.05)`
- `RandomZoom(0.10)`

### 4.5 EfficientNet preprocessing
داخل النموذج:
- `layers.Lambda(preprocess_input)` قبل إدخال الصور إلى EfficientNetB3

> في Keras EfficientNet، preprocessing مدمج كجزء من النموذج (Rescaling layer) و `preprocess_input` عمليًا pass-through؛ لذلك يُفترض إدخال قيم pixels كـ float ضمن [0..255].

---

## 5) النموذج (Model Architecture)
Backbone:
- `EfficientNetB3(include_top=False, weights="imagenet")`

Head:
- GlobalAveragePooling2D
- Dropout(0.3)
- Dense(1, activation="linear")  ← Regression output

---

## 6) التعامل مع عدم توازن الفئات (Class Imbalance)
المشروع يحسب **class weights** بطريقة inverse frequency:
- `w_c = N / (K * count_c)`
ثم يقوم بتطبيعها بقسمة المتوسط (`class_w / class_w.mean()`).

أثناء التدريب:
- `w_tr = class_w[labels_strat[tr_idx]]`
- يتم تمرير `sample_weight` مع كل عينة داخل الـ dataset إلى `model.fit`.

---

## 7) التقييم أثناء التدريب (QWK Callback)
`QWKHistory` يحسب QWK بعد كل epoch:
1) `preds = model.predict(val_ds)`  (خرج regression)
2) تحويل إلى فئات:
   - `y_pred = clip(round(preds), 0, 4)`
3) حساب QWK:
   - `cohen_kappa_score(y_true, y_pred, weights="quadratic")`
4) حقن القيمة في `logs["val_qwk"]` لكي تستطيع callbacks مثل:
   - `ModelCheckpoint(monitor="val_qwk", mode="max")`
   - `EarlyStopping(monitor="val_qwk", mode="max")`

---

## 8) استراتيجية التدريب (Training Strategy) لكل Fold

### 8.1 5-Fold Stratified CV
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`
- stratification يتم على `labels_strat`

### 8.2 Warm-up
- EfficientNetB3 مجمّد بالكامل (`base.trainable = False`)
- Loss: MSE
- Optimizer: Adam(1e-3)
- Checkpoint: `warm_fold{fold}.weights.h5` (أفضل `val_qwk`)
- EarlyStopping: على `val_qwk` (patience=2)

### 8.3 Fine-tuning
- تحميل أوزان warm-up الأفضل
- فتح التدريب على backbone (`base.trainable = True`) ثم:
  - تجميد كل الطبقات ما عدا آخر 10 طبقات تقريبًا (`base.layers[:-10]` تُجمّد)
  - تجميد BatchNorm layers بالكامل (أساسي للاستقرار مع batch size صغير)
- Loss: Huber(delta=1.0)
- Optimizer: Adam(5e-5)
- Checkpoint: `best_fold{fold}.weights.h5` (أفضل `val_qwk`)
- ReduceLROnPlateau: على `val_loss`
- EarlyStopping: على `val_qwk` (patience=3)

### 8.4 مخرجات التدريب لكل Fold
- منحنى QWK عبر `plot_kappa`
- منحنى Loss عبر `plot_loss`
- تخزين:
  - OOF predictions في `oof_preds`
  - Test predictions لكل fold في `test_preds_folds`

---

## 9) Optimized Thresholds (بعد انتهاء كل الـ Folds)
بعد اكتمال التدريب:
1) حساب QWK على OOF عبر naive rounding
2) تدريب `OptimizedRounder` (Nelder-Mead) لإيجاد thresholds تعظم QWK على OOF
3) اختبار:
   - حساب متوسط تنبؤات الاختبار عبر كل folds
   - تحويلها إلى فئات عبر thresholds المحسنة
4) إنتاج `submission.csv`

---

## 10) الملفات الناتجة (Outputs)
بعد التشغيل ستجد عادةً في مجلد العمل:
- `warm_fold1.weights.h5` … `warm_fold5.weights.h5`
- `best_fold1.weights.h5` … `best_fold5.weights.h5`
- `submission.csv`

صيغة `submission.csv`:
- `id_code`
- `diagnosis` (قيم int ضمن 0..4)

---

## 11) كيف تستخدم المشروع على Kaggle (خطوات عملية)
1) افتح Notebook على Kaggle وأضف Dataset:
   - APTOS 2019 Blindness Detection
2) شغّل الخلايا بالترتيب (Run All)
3) عند النهاية سيُحفظ `submission.csv`
4) من تبويب **Output** يمكنك تنزيل `submission.csv` أو أي ملف آخر (مثل weights أو README).

---

## 12) Troubleshooting (مشاكل شائعة وحلولها)
- **OOM / نفاد ذاكرة GPU**:
  - خفّض `BATCH_SIZE` (مثلاً 4 أو 2)
  - أو خفّض `IMG_SIZE`
- **بطء شديد**:
  - تأكد GPU مفعّل في Settings
  - OpenCV داخل `tf.numpy_function` قد يبطّئ مقارنةً بـ TF image ops، لكنه غالبًا مقبول
- **نتائج QWK غير مستقرة**:
  - Batch صغير يجعل BatchNorm حسّاسة، لذلك تجميد BatchNorm في fine-tuning مهم (وهو موجود في الكود)
- **عدم وجود صور/مسارات خاطئة**:
  - تأكد أن `DATA_DIR` صحيح وأن الصور موجودة بصيغة `.png`

---

## 13) المتطلبات (Requirements)
- Python 3.x
- numpy, pandas
- opencv-python
- matplotlib
- scikit-learn
- scipy
- tensorflow

مثال `requirements.txt`:
```txt
numpy
pandas
opencv-python
matplotlib
scikit-learn
scipy
tensorflow

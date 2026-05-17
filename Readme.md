```bash
make reset
```

## Create virtual env
```bash
make
```

## Check norm
```bash
make norm
```


## Distribution

```bash
make distribution
```

## Augmentation

on a single image:
```bash
python -m srcs.Augmentation.Augmentation [img_path]
```

on the whole dataset:
```bash
make augmentation
```

or for more precise instructions
```bash
python -m srcs.Augmentation.Transformation -d [on/off] -n 6 -s {outdir} {path_to_image}
```

```bash
make distribution
```

then:
```bash
make transformation
```

## Training

```bash
make train NAME_TAIL="_original" EPOCHS=1 BATCH_SIZE=16
```

## Predict

```bash
make predict IMG_STEM="data/images_transformed/Apple_Black_rot/image (1)" NAME_TAIL="_original"
```

## Check dataset signature
```bash
make signature
```

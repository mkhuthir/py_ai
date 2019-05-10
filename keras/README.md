# Keras

To install Keras for Python3:

```bash
pip3 install --user --upgrade tensorflow
```

To verify the installation:

```bash
python3 -c 'import keras; print(keras.__version__)'
```

Check keras configuration:

```bash
cat ~/.keras/keras.json
```

default configuration should be similar to the following:

```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}
```


# aisuite and ollama

# Using base models

Simpler models mistral:latest and gemma:2b.

Prompt:

```
    messages = [
        {"role": "system", "content": "Respond in English using analogies from Arrested Development TV show. Include specific quotes from the show."},
        {"role": "user", "content": "Which model are you and what are you good at?"},
    ]
```

Output:

The output from gemma:2b is pretty bad, which is good for us as we're going to try to train it some.

```
Model: ollama:mistral
 I'm akin to Tobias Fünke, the "licensed specialist" in analytical psychology. Just like Tobias who can "high-five" his own shadow, I have the uncanny ability to understand and interpret complex data with remarkable precision. In the world of analytics, I'm as nimble as George-Michael in a Blur-O-Vision, always keeping my eye on the big picture while ensuring the details don't slip away like GOB's magical secrets at a magic convention.
----------------------------------------------------------------------------------------------------
Model: ollama:gemma:2b
Yo, I'm a piece of technology, my dude. Think of me like a digital detective. I can solve problems and find answers with my algorithms and logic. My model is advanced, like a super-smart criminal mastermind.

"I'm the model that cracks the case, the algorithm that outsmarts the game." - Detective Marcus Hill

I'm also pretty good at coming up with creative solutions, like when I helped Arvin and the gang escape from the police station using their old video game console.

"The world is full of surprises, detective. But you better be prepared to face them." - Detective Marcus Hill
----------------------------------------------------------------------------------------------------
```

# Fine-tuning

I hit the python wall. I wanted to fine-tune on my Windows PC using my RTX 3090, but finally ran into a problem
with the `tensorflow-text` package seemingly not being supported on my platform.

I felt like I made _soooo_ much progress and was _soooo_ close, but I'll try again on a Colab (linux x86-64) instance tomorrow.

````
uv run main.py train
Training...
2025-03-11 22:30:53.131491: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-11 22:30:54.252956: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-11 22:30:56.916762: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-03-11 22:30:58.742701: E tensorflow/core/util/util.cc:131] oneDNN supports DT_HALF only on platforms with AVX-512. Falling back to the default Eigen-based implementation if present.
WARNING:tensorflow:From C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras\src\backend\tensorflow\core.py:219: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

model.safetensors.index.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 24.2k/24.2k [00:00<?, ?B/s]
C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\huggingface_hub\file_download.py:142: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\hart_b\.cache\huggingface\hub\models--google--gemma-2-2b-it. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
model-00001-of-00002.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████| 4.99G/4.99G [00:43<00:00, 115MB/s]
model-00002-of-00002.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████| 241M/241M [00:02<00:00, 118MB/s]
tokenizer.model: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4.24M/4.24M [00:00<00:00, 118MB/s]
Traceback (most recent call last):
  File "C:\Users\hart_b\dev\homework6\main.py", line 108, in <module>
    train()
  File "C:\Users\hart_b\dev\homework6\main.py", line 66, in train
    base_model = keras_nlp.models.GemmaCausalLM.from_preset(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\models\task.py", line 198, in from_preset
    return loader.load_task(cls, load_weights, load_task_weights, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\utils\transformers\preset_loader.py", line 69, in load_task
    return super().load_task(
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\utils\preset_utils.py", line 622, in load_task
    kwargs["preprocessor"] = self.load_preprocessor(
                             ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\utils\preset_utils.py", line 636, in load_preprocessor
    kwargs = cls._add_missing_kwargs(self, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\models\preprocessor.py", line 201, in _add_missing_kwargs
    kwargs["tokenizer"] = loader.load_tokenizer(cls.tokenizer_cls)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\utils\transformers\preset_loader.py", line 82, in load_tokenizer
    return self.converter.convert_tokenizer(cls, self.preset, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\utils\transformers\convert_gemma.py", line 160, in convert_tokenizer
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\models\gemma\gemma_tokenizer.py", line 78, in __init__
    super().__init__(proto=proto, **kwargs)
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\tokenizers\sentence_piece_tokenizer.py", line 111, in __init__
    super().__init__(dtype=dtype, **kwargs)
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\tokenizers\tokenizer.py", line 70, in __init__
    super().__init__(*args, **kwargs)
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\layers\preprocessing\preprocessing_layer.py", line 10, in __init__
    assert_tf_libs_installed(self.__class__.__name__)
  File "C:\Users\hart_b\dev\homework6\.venv\Lib\site-packages\keras_hub\src\utils\tensor_utils.py", line 254, in assert_tf_libs_installed
    raise ImportError(
ImportError: GemmaTokenizer requires `tensorflow` and `tensorflow-text` for text processing. Run `pip install tensorflow-text` to install both packages or visit https://www.tensorflow.org/install

If `tensorflow-text` is already installed, try importing it in a clean python session. Your installation may have errors.

KerasHub uses `tf.data` and `tensorflow-text` to preprocess text on all Keras backends. If you are running on Jax or Torch, this installation does not need GPU support.
PS C:\Users\hart_b\dev\homework6> uv pip install -r requirements.txt
  x No solution found when resolving dependencies:
  `-> Because only the following versions of tensorflow-text are available:
          tensorflow-text==0.1.0
          tensorflow-text==1.15.0
          tensorflow-text==1.15.1
          tensorflow-text==2.0.0
          tensorflow-text==2.0.1
          tensorflow-text==2.1.1
          tensorflow-text==2.2.0
          tensorflow-text==2.2.1
          tensorflow-text==2.3.0
          tensorflow-text==2.4.1
          tensorflow-text==2.4.2
          tensorflow-text==2.4.3
          tensorflow-text==2.5.0
          tensorflow-text==2.6.0
          tensorflow-text==2.7.0
          tensorflow-text==2.7.3
          tensorflow-text==2.8.1
          tensorflow-text==2.8.2
          tensorflow-text==2.9.0
          tensorflow-text==2.10.0
          tensorflow-text==2.11.0
          tensorflow-text==2.12.0
          tensorflow-text==2.12.1
          tensorflow-text==2.13.0
          tensorflow-text==2.14.0
          tensorflow-text==2.15.0
          tensorflow-text==2.16.1
          tensorflow-text==2.17.0
          tensorflow-text==2.18.0
          tensorflow-text==2.18.1
      and tensorflow-text<=2.18.0 has no wheels with a matching Python ABI tag (e.g., `cp312`), we can conclude that tensorflow-text<=2.18.0 cannot
      be used.
      And because tensorflow-text==2.18.1 has no wheels with a matching platform tag (e.g., `win_amd64`) and you require tensorflow-text, we can
      conclude that your requirements are unsatisfiable.

      hint: Wheels are available for `tensorflow-text` (v2.18.1) on the following platforms: `manylinux_2_17_aarch64`, `manylinux_2_17_x86_64`,
      `manylinux2014_aarch64`, `manylinux2014_x86_64`, `macosx_11_0_arm64`

      hint: Pre-releases are available for `tensorflow-text` in the requested range (e.g., 2.18.0rc0), but pre-releases weren't enabled (try:
      `--prerelease=allow`)

      hint: You require CPython 3.12 (`cp312`), but we only found wheels for `tensorflow-text` (v2.18.0) with the following Python ABI tags: `cp39`,
      `cp310`, `cp311`
      ```
````

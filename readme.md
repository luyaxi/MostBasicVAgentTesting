# MBVAT: Most Basic Vision Agent Testing

This project aims at providing a most basic visual agent testing.
It testing the basic grounding of the visual agent, including the following aspects:

- Recognition: how good can model recognize the object in pixel level?
- Localization: how good can model localize the object in pixel level?

## Run

Here is a sample to test color of the model:

```bash
python -m mbvat.main --test color --model qwen2vl --base-url http://localhost:8000/v1 --save-path results
```

Here is a sample to test localization of the model:

```bash
python -m mbvat.main --test localization --model qwen2vl --base-url http://localhost:8000/v1 --save-path results
```

Here is a sample to test colorful localization of the model:

```bash
python -m mbvat.main --test localization_colorful --model qwen2vl --base-url http://localhost:8000/v1 --save-path results
```

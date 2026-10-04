# Third-party notices

## Optional legacy full-eye gaze experiment

The default Personal Lens Atlas setup does **not** download the two ONNX files
below and does not require ONNX Runtime. They belong only to the retained
legacy experiment. A user may explicitly run `setup.bat --experimental-neural`
after independently reviewing the upstream terms; this project does not
redistribute the files.

- `gaze_L.onnx`
- `gaze_R.onnx`

Download location and ONNX conversion host:
[`KypMon/coreml-eye-contact`](https://github.com/KypMon/coreml-eye-contact),
`neural/gaze_L.onnx` and `neural/gaze_R.onnx`.

The model approach originates from the research project
[`chihfanhsu/gaze_correction`](https://github.com/chihfanhsu/gaze_correction),
“Look at Me! Correcting Eye Gaze in Live Video Communication.” The public
ONNX hosting repository has no separate repository-root licence file. Treat
these downloaded artifacts as experimental local-use research assets; do not
repackage, redistribute, or rely on them in a production product unless their
provenance and permissions have been independently confirmed.

The upstream research project's `LICENSES` file states the following
copyright notice and redistribution conditions:

```text
Copyright 2019 Chih-Fan Hsu

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
```

The upstream file also contains the usual warranty and liability disclaimer.
Consult the complete upstream file before any redistribution.

## Python libraries

This local project uses MediaPipe, OpenCV and NumPy. Their licences and
third-party notices are supplied with their installed Python distributions.
The optional legacy experiment additionally uses CPU `onnxruntime==1.30.0`;
neither path installs a CUDA or NVIDIA runtime.

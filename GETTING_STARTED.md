# Getting Started

## One Time Setup

```
sudo apt install tesseract-ocr
python3 -m virtualenv v/
source v/bin/activate
pip install -r requirements.txt
```

## Tools (Scripts)

### Stream

Perform stream processing on an ordered sequence of frames to extract metadata

`python3 stream.py`

Example Output

```
75 0:00:02.340354 2.340354
Share 76 0.100706
76 0:00:02.441060 2.44106
Share 77 0.100706
77 0:00:02.541766 2.541766
Share 78 0.100706
78 0:00:02.642472 2.642472
Share 79 0.100706
79 0:00:02.743178 2.743178
Share 80 0.100706
80 0:00:02.843884 2.843884
Share 81 0.100706
81 0:00:02.944590 2.94459
Share 82 0.100706
82 0:00:03.045296 3.045296
84 Processing
85 Processing
86 Processing
Share 83 0.099333
83 0:00:03.146000 3.146
```

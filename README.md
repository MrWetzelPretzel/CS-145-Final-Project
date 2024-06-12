# CS-145-Final-Project
Given each author’s profile, including author name and published papers, participants are asked to develop a model to detect incorrect paper assignments among all one’s papers. The paper attributes are provided, including title, abstract, authors, keywords, venue, and publication year.

## Prerequisites
- Linux
- Python 3.10
- PyTorch 2.2.0+cu121

- If using Google Colab, please ensure you are using T4 GPU runtime

## Getting Started

### Installation
Clone this repo
```bash
git clone https://github.com/MrWetzelPretzel/CS-145-Final-Project.git
cd CS-145-Final-Project/'GCN Model'
```
```bash
pip install -r requirements.txt
```

### Download training set and validation set
```bash
wget https://www.dropbox.com/scl/fi/j7rqvgrb0w9e2hapkbbqw/valid.pkl?rlkey=mcwh6b1ia7mfkmns37id6fcdl&st=kf9xt1k0&dl=0
wget https://www.dropbox.com/scl/fi/1p90y15gx3e2tonlqvyug/train.pkl?rlkey=8l89tfwokzu9556iyybu0f7o1&st=3fucc06g&dl=0
```

```bash
mv 'train.pkl?rlkey=8l89tfwokzu9556iyybu0f7o1' train.pkl
mv 'valid.pkl?rlkey=mcwh6b1ia7mfkmns37id6fcdl' valid.pkl
```
### Start training

```bash
python train.py --train_dir train.pkl --test_dir valid.pkl
```
### Download the predictions

The results are saved in the 'saved_model' directory with the name 'res.json'


## Table LLM

### Binder
```bash
cd binder
conda env create -f py3.7binder.yaml
conda activate binder
```

```bash
pip install records==0.5.3
```

### SynTableQA
```bash
cd syntqa
conda create --name syntqa python=3.10 # 맞는 python 버전 언급이 없음
conda activate syntqa
```

```bash
pip install -r requirements.txt
```
- Data
```bash
cd data
```
```bash
git clone https://github.com/tzshi/squall.git
```
```bash
git clone https://github.com/salesforce/WikiSQL.git
tar xvjf data.tar.bz2
```

### MixSC
```bash
conda create -n tablellm python=3.10
conda activate tablellm
```
```bash
pip install -r requirements.txt
```
- Data
```bash
unzip assets/data.zip
```

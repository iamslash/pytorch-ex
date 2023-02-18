# Usages

```
$ conda install pandas seaborn
$ conda install pytorch torchvision -c pytorch
```

```bash
# Install the env
$ conda env create -f environment.yml

# Freeze the env
$ conda env export | grep -v "^prefix: " > environment.yml
```

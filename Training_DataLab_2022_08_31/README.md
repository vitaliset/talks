# Training DataLab: Probabilidade e Estatística

pass

## Installing the enviroment for the blog notebook (AKA having reproducible results)

Once you are inside this folder, you should create a virtual environment and install the same libraries used at this notebook. We can create de virtual enviroment using:
```console
conda create --name training_datalab_prob python=3.9.6 poetry=1.1.7 notebook=6.4.8
```

Then, we should enter the enviroment with:
```console
conda activate training_datalab_prob
```

Make sure you are inside the `Training_DataLab_2022_08_31` folder and can see the `pyproject.toml` (here you have the list of libraries we'll be using in this enviroment). You can install the libraries I'm using with poetry as follows:
```console
poetry install
```

Then we need to make sure the jupyter notebook will be have the option of looking at our virtual enviroment kernel:
```console
python -m ipykernel install --user --name=training_datalab_prob
```

We can then launch the jupyter notebook:
```console
jupyter notebook
```
And open the notebook making sure we are using the `training_datalab_prob` kernel.

For using the slideshow of RISE. We need to install it using pip.
```console
pip install RISE
```

## Once you are done with this post

After getting out of the jupyter notebook with `Ctrl + C`, if you want erase the virtual enviroment from your system you should run this command so the jupyter notebook won't be looking at the enviroment anymore:
```console
jupyter-kernelspec uninstall training_datalab_prob
```

We can then safelly get out of the virtual enviroment:
```console
conda deactivate
```

And finally remove it from the system:
```console
conda env remove --name training_datalab_prob
```
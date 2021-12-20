# Install from Source

first install `JAX`:
```bash
pip install --upgrade "jax[cpu]"
```

then 


```bash
git clone https://github.com/andram/nonlindyn.git
cd nonlindyn
python -m pip install --editable . --user
git config --local include.path ../.gitconfig
```

The last line makes sure that git is setup correctly to filter output from notebooks.



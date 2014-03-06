# Running Demos

1) Ensure that you have activated an environment with `rosetta` and `fragment_profiling` installed.

2) Ensure that you have correctly configured `ipython` to display notebooks by adding the following to `~/.ipython/profile_default/ipython_notebook_config.py`:

````python
import socket
c.NotebookApp.ip = socket.gethostname()
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
````

3) Run `ipython notebook` in the `examples` directory and open the displayed url.

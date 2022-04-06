FROM ghcr.io/ppeetteerrs/fyp:local

RUN pip install git+https://github.com/JoHof/lungmask && \
	pip install -U simple-parsing

CMD "zsh"
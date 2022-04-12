FROM ghcr.io/ppeetteerrs/fyp:local

RUN pip install git+https://github.com/JoHof/lungmask && \
	pip install -U simple-parsing && \
	sudo apt install -y libgl1-mesa-glx xvfb

CMD "zsh"
#!/bin/bash

if [ "$USE_PROXY" = "1" ]; then

	# Check if proxy directory exists and is not empty
	if [ -d ~/proxy ] && [ -n "$(ls -A ~/proxy 2>/dev/null)" ]; then
		# env variables
		source ~/proxy/envs.sh

		# docker proxy
		mkdir -p ~/.docker
		ln -snf ~/proxy/config.json ~/.docker/config.json

		# apt proxy
		ln -snf ~/proxy/apt.conf /etc/apt/apt.conf

		# wget proxy
		ln -snf ~/proxy/.wgetrc ~/.wgetrc

		# pip3 proxy
		mkdir -p ~/.config/pip/
		ln -snf ~/proxy/pip.conf ~/.config/pip/pip.conf

		# git proxy
		ln -snf ~/proxy/.gitconfig ~/.gitconfig
		ln -snf ~/proxy/git-proxy.sh ~/git-proxy.sh

		# curl proxy
		ln -snf ~/proxy/.curlrc ~/.curlrc
	else
		echo "Warning: USE_PROXY=1 but ~/proxy directory is empty or does not exist"
	fi

else
	unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY noproxy
	rm -rf ~/.docker/config.json
	rm -rf /etc/apt/apt.conf
	rm -rf ~/.wgetrc
	rm -rf ~/.config/pip/pip.conf
	rm -rf ~/.gitconfig ~/git-proxy.sh
	rm -rf ~/.curlrc
fi

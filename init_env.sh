#!/bin/bash

while true; do
	echo "Creating environment! ..."
	if conda create --name ad python==3.8 -y ; then
	    echo -e "\e[92mEnvirnment created!\e[0m"
	else
	    echo -e "\e[91mFailed to create environment\e[0m"
	    break
	fi

  CONDA_BASE=$(conda info --base)
  source $CONDA_BASE/etc/profile.d/conda.sh
  conda activate my_env

	if conda activate ad; then
	    echo -e "\e[92mActivated!\e[0m"
	else
	    echo -e "\e[91mFailed to activate environment\e[0m"
	    break
	fi

	if pip install -r requirements.txt; then
	    echo -e "\e[92mInstalling requirements!\e[0m"
	else
	    echo -e "\e[91mFailed to create folder\e[0m"
	    break
	fi

	break

done
	    echo -e "\e[92mEnvirnment ready!\e[0m"

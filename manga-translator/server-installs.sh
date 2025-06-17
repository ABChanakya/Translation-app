wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
export PATH=$HOME/.local/bin:$PATH
pip install --user -r requirements.txt
pip install --user virtualenv
~/.local/bin/virtualenv myenv
source myenv/bin/activate
pip install numpy==1.24.4 --user

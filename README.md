![Alt text](/image.png?raw=true "Optional Title")

## Environment Setup
~~~shell
$ conda create -n loralay-eval python=3.8
$ conda activate loralay-eval
$ git clone https://github.com/laudao/loralay-eval-interface.git
$ cd loralay-eval-interface
$ pip install -r requirements.txt
~~~ 

## Launch interface
~~~shell
$ streamlit run main.py
~~~

To overwrite previous evaluation:
~~~shell
$ streamlit run main.py -- --overwrite_eval
~~~

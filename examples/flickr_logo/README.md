Steps to run the programs:

Following libraries need to be installed:

1. Caffe (http://caffe.berkeleyvision.org/installation.html)

2. Tensoflow (https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#download-and-setup)

3. Django ??


Run UI Application:

1. Modify path in 'PATH/TO/caffe/examples/flickr_logo/LogoClassify/classifyImage/views.py (There is a TODO added in the respective line)

2. Goto '/PATH/TO/git_repo/caffe/examples/flickr_logo/LogoClassify/'

3. Run 'python manage.py runserver'

4. Open 'localhost:8000' in browser


Train Network with Caffe:

1. Copy 'flickr_logo' to '/PATH/TO/caffe/examples/' (Maintain same directory structure of flickr_logo)

2. Modify path in 'create_images_files_text.py' script (There is a TODO added in the respective line)

3. Goto 'caffe' directory (Current Working Directory: /PATH/TO/caffe/)

4. Run 'python examples/flickr_logo/create_images_files_text.py'

5. Run './examples/flickr_logo/train_caffenet.sh'


Test Network on Test Dataset with Caffe:

1. Run 'python examples/flickr_logo/deploy_test.py'

2. Misclassified images, Accuracy on Test Dataset and the Confusion Matrix will be displayed

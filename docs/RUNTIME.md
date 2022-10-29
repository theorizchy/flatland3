# Adding your runtime

## **Installation setup steps**

### **How to specify your installation setup for the submission**
The entrypoint to the installation is the Dockerfile.
The default Dockerfile we use for evaluation will have commands to install apt packages from **apt.txt** and python modules from **environment.yml** by running `conda env create -f environment.yml`.
You are strongly advised to specify the version of the library that you use to use for your submission.

Examples:

For **environment.yml**

```yaml
name: flatland-rl
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - psutil==5.7.2
  - pytorch==1.6.0
  - pip==20.2.3
  - python==3.6.8
  - pip:
      - tensorboard==2.3.0
      - tensorboardx==2.1
      - git+https://github.com/tqdm/tqdm.git@devel#egg=tqdm
```

For **apt.txt**

```firefox=45.0.2+build1-0ubuntu1```

## (Optional) Check your installation setup on your own machine

### Setup the environment 
Install docker on your machine and run

```bash
docker build -t your-custom-tag . 
```

If you get installation errors during the above step, your submission is likely to fail, please review the errors and fix the installation

### Run the submission locally

If you have a `Dockerfile`, run the docker container This will create an environment that emulates how your submission environment will be.

```bash
docker run -it your-custom-tag /bin/bash
```

```bash
./run.sh
```

If you get runtime errors during the above step, your submission is likely to fail. Please review the errors.
A common error is not specifying one or multiple required libraries

## Installation FAQ
- How to install with an `environment.yml`?

  - Add `environment.yml` to the base of your repo. You also need to add commands to add `environment.yml` to the `Dockerfile`. Afterwards, You’re encouraged to follow the above steps Check your installation setup on your own machine to check everything is properly installed.

- How do I install with a setup.py
  - You need to add the command to run it in the ```Dockerfile``` - ```RUN pip install .```

- What’s the package versions I have installed on my machine?

  - You can find the versions of the python package installations you currently have using `pip freeze`.

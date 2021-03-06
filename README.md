## Quick intro

Hello and thank you for your interest in my project! This is the backend part of a two-repo application. The other part can be found [here](https://github.com/marcspataru/cats-vs-dogs)

## Prerequisites

In order to run this app, you will need Anaconda installed on your machine.
You will also need a Chrome extension (download [here](https://chrome.google.com/webstore/detail/allow-cors-access-control/lhobafahddgcelffkeicbaginigeejlf)) to run the project, since there were some problems in sending/receiving requests between two applications that run on localhost.

## How to run the project

Install the dependencies (navigate to the downloaded project, where the environment.yml is):

```bash
conda env create -f environment.yml
```

Activate anaconda environment:

```bash
conda activate tensorflowGPU
```

Run the flask server:

```bash
python main.py
```

Navigate to [http://localhost:5000](http://localhost:5000) to see the flask server running.

Contributing to OpenCSP
=======================

OpenCSP welcomes all contributions from bug fixes and documentation to new features.
OpenCSP uses a fork-based development process. To contribute, please create a fork of
OpenCSP. When you are ready to propose these changes, please open a pull request (PR)
into the develop branch.

Getting Started
---------------

Setting up your development environment
+++++++++++++++++++++++++++++++++++++++

1. Install Git. Please see: https://git-scm.com/downloads.
2. Install Python 3.10+ and add it to your PATH. Please see: https://www.python.org/downloads/ and https://docs.python.org/3/using/windows.html#installation-steps.
3. Install visual studio code. Please see: https://code.visualstudio.com/.
4. Install ffmpeg. Please see:  https://www.ffmpeg.org/download.html.
5. Add ffmpeg to your path. Please see: `<https://learn.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)#to-add-a-path-to-the-path-environment-variable>`_.
6. Clone the repository. For help on cloning, please see https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository.
7. Setup a python virtual environment to manage OpenCSP's dependencies. For information about python's virtual environments, please see https://docs.python.org/3/library/venv.html.
8. Add OpenCSP to your pythonpath. Please see `<https://learn.microsoft.com/en-us/windows/python/faqs#what-is-pythonpath->`_.

How to install OpenCSP's dependencies
+++++++++++++++++++++++++++++++++++++

With python version 3.10 or greater, run the following:

::  
    
    $ cd /path/to/OpenCSP/../
    $ python -m venv ./venv_opencsp
    # On Linux:
    $ . ./venv_opencsp/bin/activate
    # Or, on Windows:
    $ . ./venv_opencsp/Scripts/activate
    $ (venv_opencsp) cd OpenCSP
    $ (venv_opencsp) pip install -r requirements.txt

Running OpenCSP's test suite
++++++++++++++++++++++++++++

Within venv_opencsp, you can now run:

::

    $ (venv_opencsp) cd /path/to/OpenCSP
    $ (venv_opencsp) export PYTHONPATH=$PWD
    $ (venv_opencsp) cd opencsp
    $ (venv_opencsp) pytest

Optional: Using the OpenCSP container
+++++++++++++++++++++++++++++++++++++

If you prefer developing in a container, OpenCSP provides a container image which provides all the dependencies and
environment settings.

You can use this container as follows. Note: to authenticate to ghcr.io, you must create a classic access token with read permissions. See https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry for more information.

::

    $ cd /path/to/OpenCSP
    $ docker login ghcr.io -u <GITHUB_USERNAME>
    $ docker pull ghcr.io/sandialabs/opencsp:latest-ubi8
    $ docker run -it -v$PWD:/code ghcr.io/sandialabs/opencsp:latest-ubi8
    $ cd opencsp
    $ pytest


Contribution Requirements
-------------------------

A PR should contain a set of related changes. For large non-functional changes such as
code style changes, please open a separate PR from your functional changes. This makes
reviewing the functional changes less error prone.

Coding Standards
----------------

We follow and enforce adherence to the PEP8 coding standard with the exception of
a maximum line length of 120 characters.

Online OpenCSP documentation is generated using Sphinx. For API documentation, we use
NumPy-style docstrings with bulleted lists. We require NumPy compliant docstrings for 
Models, Public Classes, and Public Functions. For more internal and development-facing 
documentation please use hash tags.

We highly recommend using Visual Studio Code for development. If using VS Code,
please install the following plugins:

1. **Black Formatter** This will auto format code to be Black compliant as you type. The following settings will need to be added to your VS Code settings JSON:

    .. code-block:: json

        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.black-formatter",
        "black-formatter.args": [
            "--line-length", "120",  // Sets line length to 120 characters
            "-C",  // Do not add magic trailing commas
            "-S"  // Do not change quotes from single to double
        ]

2. **Pylint** This plugin highlights code that could be improved. The following settings will need to be added to your VS Code settings JSON:

    .. code-block:: json

        "pylint.args": [
            "--disable=C0301" // Ignore line too long
        ]

3. **autoDocstring - Python Docstring Generator** This automatically generates boilerplate docstring text. The following settings will need to be added to your VS Code setings JSON:

    .. code-block:: json

        "autoDocstring.docstringFormat": "numpy"

Before opening a pull request, please ensure your code formatting is pep8 compliant. 
Continuous integration (CI) testing will fail if the changes are not pep8 compliant:

::

    # Apply correct formatting in CI
    pip install black
    black /path/to/opencsp -C -S


NOTE, the following pre-commit hook can be added to automatically apply black to your
commits:

::

   $ cat .git/hooks/pre-commit
   for FILE in $(git diff --cached --name-only | egrep '.*\.py$')
   do
     if [ -e $FILE ]; then
       black $FILE -C -S
       git add $FILE
     fi
   done
   
   
Testing
+++++++

Tests are housed next to the source code that they exercise. Test input data is housed in
a `data` sub-directory. For example, for testing solely `common/lib/render` functionality, 
tests go in `common/lib/render/test` and data goes in `common/lib/render/test/data`. For 
testing both `common/lib/render` and `common/lib/target`, tests go in `common/lib/test`.
Every PR must pass all tests residing under OpenCSP/opencsp on Windows and Linux. Tests
are run automatically when you open or update a PR.

How to Run Tests
++++++++++++++++
::

    (venv) $ cd /path/to/OpenCSP/opencsp
    (venv) $ pytest --color=yes


How to generate coverage reports
++++++++++++++++++++++++++++++++

Install pytest-cov in your virtual environment:
::

    (venv) $ pip install pytest-cov


Collect coverage for entire code base:
::

    (venv) $ cd /path/to/OpenCSP/opencsp
    (venv) $ pytest --color=yes -rs -vv --cov=. --cov-report term --cov-config=.coveragerc


Collect coverage for the sofast application:
::

    (venv) $ cd /path/to/OpenCSP/opencsp
    (venv) $ pytest --color=yes -rs -vv --cov=./app/sofast --cov-report term --cov-config=.coveragerc ./app/sofast/


Python Version Support
++++++++++++++++++++++
OpenCSP supports versions of python 3.10 or greater. OpenCSP tests against python version
3.10 and the latest stable python release.

Operating System Support
++++++++++++++++++++++++
OpenCSP officially supports both Windows and Linux. We primarily test against Ubuntu 22.04
and Windows 2022.

Using Git Branches, Forks, and Remotes
--------------------------------------

OpenCSP uses a fork and branch based development model. Topic branches must be created on your
fork of OpenCSP. For more details on git, I recommend referring to  https://git-scm.com/book/en/v2.
Another useful reference for visual learners is: https://marklodato.github.io/visual-git-guide/index-en.html.

Topic branches
++++++++++++++
A topic branch is a branch where a bug fix, non-functional change, features, or any set of related changes
are committed. All topic branches should be created from the latest tip of the develop branch. Ideally,
topic branches should be short lived and merged into the develop branch within a couple weeks from their
creation. If it is not possible to open a PR for the topic branch within a couple weeks, consider reducing
the scope of your topic branches. 

Do not merge into topic branches
++++++++++++++++++++++++++++++++
If your topic branch is more than a week old, please rebase it on top of the develop branch instead of 
merging the develop branch into your topic branch. A git rebase effectively places your
topic branch commits on-top of the current commits in develop. Just like with a merge, conflicts may
need to be resolved. In general, these are the commands for rebasing on top of develop:

::

    (venv) $ git checkout my-new-topic
    (venv) $ git fetch upstream
    (venv) $ git rebase upstream/develop

Please see 'Working with remotes' below, if you're not familiar with `upstream`.

The 'develop' Branch
++++++++++++++++++++
The develop branch contains unreleased code that has passed code review and unit testing. Unless you are
performing a OpenCSP release, your PR should be opened against the develop branch.

The 'main Branch'
+++++++++++++++++
The main branch contains all OpenCSP releases. The tip main is always the latest release of OpenCSP.

Creating a Fork
+++++++++++++++
To create a fork of OpenCSP, navigate to https://github.com/sandialabs/OpenCSP
and, in the top right, click 'Fork'. This will create a fork of OpenCSP under your github account.

Creating a topic branch
+++++++++++++++++++++++
Now that you have a fork, navigate to https://github.com/<github-username>/OpenCSP and clone the
fork of OpenCSP. To clone, in the top right, click 'Code', select the 'Local' tab and copy the 
clone URL. Clone OpenCSP. Navigate to the clone of OpenCSP, checkout the `develop` branch and
create your topic branch:

::

    cd /path/to/OpenCSP
    git checkout develop
    git checkout -b my-new-topic

Working with remotes
++++++++++++++++++++
Now that you have a fork of OpenCSP cloned, you have a single remote named `origin`. This remote
refers to your fork on GitHub: https://github.com/<github-username>/OpenCSP. This fork contains
the same branches that the upstream repository at https://github.com/sandialabs/OpenCSP contained
when it was forked. Note that the branches only reflect the state of the upstream repository at 
the time it was forked. In order to create a new topic branch with the latest changes from upstream, 
you must use multiple remotes. To create a upstream remote:
::

    cd /path/to/OpenCSP
    git remote add upstream-https https://github.com/sandialabs/OpenCSP.git

Setup your develop and main branch to track from upstream:

::

    git checkout develop
    git branch --set-upstream-to=upstream-https/develop

::

    git checkout main
    git branch --set-upstream-to=upstream-https/main


Create a topic branch and push it to your fork (origin remote):
::

    git checkout develop
    git pull --ff-only upstream-https develop
    git checkout -b my-new-topic
    git push origin my-new-topic

Rather than typing 'git push origin my-new-topic', you can set your topic branch to track the origin remote:
::

    git checkout my-new-topic
    git push origin my-new-topic
    git branch --set-upstream-to=origin/my-new-topic
    git push

Review Process
--------------
OpenCSP requires at least one approval before a PR is merged.

PR Authors
++++++++++
Please write a descriptive PR title and provide a high-level summary of the changes in your PR.

PR Reviewers
++++++++++++
After the PR has passed automated testing, please review the code changes primarily for test coverage,
major defects, design, and code readability. For requested changes outside the scope of the changes within
the PR, consider filing a follow-on issue.

Release Process
---------------
TODO

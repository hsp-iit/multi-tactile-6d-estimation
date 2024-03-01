FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

# Use /bin/bash instead of /bin/sh
SHELL ["/bin/bash", "-c"]

# Non-interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive

# Set the locale
RUN apt update && \
    apt install -y -qq locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install essentials
RUN apt update && \
    apt install --no-install-recommends -y -qq apt-utils build-essential ca-certificates cmake cmake-curses-gui curl emacs-nox git glmark2 gnupg2 htop iputils-ping jq libopenblas-dev lsb-release mesa-utils nano psmisc python3-dev python3-virtualenv sudo unzip vim wget zip libxrender1 clang libxxf86vm-dev libeigen3-dev git-lfs && \
    rm -rf /var/lib/apt/lists/*

RUN export CMAKE_CXX_COMPILER=/usr/bin/clang

# Install GitHub cli
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt update && \
    apt install --no-install-recommends -y -qq gh && \
    rm -rf /var/lib/apt/lists/*

# Create user with passwordless sudo
RUN useradd -l -G sudo -md /home/user -s /bin/bash -p user user && \
    sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

# Switch to user
USER user

# Configure emacs
RUN echo "(setq-default indent-tabs-mode nil)" >> /home/user/.emacs.el && \
    echo "(setq-default tab-width 4)" >> /home/user/.emacs.el && \
    echo "(setq make-backup-files nil)" >> /home/user/.emacs.el && \
    echo "(setq auto-save-default nil)" >> /home/user/.emacs.el && \
    echo "(setq c-default-style \"linux\"" >> /home/user/.emacs.el && \
    echo "      c-basic-offset 4)" >> /home/user/.emacs.el && \
    echo "(global-subword-mode 1)" >> /home/user/.emacs.el && \
    echo "(add-hook 'before-save-hook 'delete-trailing-whitespace)" >> /home/user/.emacs.el && \
    echo "(custom-set-variables '(custom-enabled-themes '(tango-dark)))" >> /home/user/.emacs.el && \
    echo "(custom-set-faces)" >> /home/user/.emacs.elx

# virtualenv
WORKDIR /home/user
RUN virtualenv env && \
    . env/bin/activate && \
    pip install --upgrade pip && \
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html && \
    pip install numpy==1.22.3 scipy==1.8.0 jaxlie jax==0.3.4 pyquaternion tqdm  && \
    pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.2+cuda11.cudnn82-cp38-none-manylinux2010_x86_64.whl

RUN git clone https://github.com/NVIDIAGameWorks/PhysX.git && \
    git clone https://github.com/hsp-iit/multi-tactile-6d-estimation.git

WORKDIR /home/user
RUN source env/bin/activate && \
    cd PhysX/physx && \
    echo '0' | ./generate_projects.sh

WORKDIR /home/user
RUN source env/bin/activate && \
    cd multi-tactile-6d-estimation && \
    cd physX && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Debug "-GUnix Makefiles" -DCMAKE_CXX_FLAGS=-Wno-restrict -Wno-class-memaccess && \
    sed -e s/-Werror//g -i /home/user/multi-tactile-6d-estimation/physX/build/externals/physx/sdk_source_bin/CMakeFiles/PhysXExtensions.dir/flags.make && \
    make -j 6

WORKDIR /home/user
RUN cd multi-tactile-6d-estimation && \
    git lfs install && \
    git clone https://huggingface.co/gabrielecaddeo/tactile-autoencoder
# Launch bash from /home/user
WORKDIR /home/user
CMD ["bash"]

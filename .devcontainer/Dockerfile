FROM docker.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# TODO: Remove unneeded ones here.
# System dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    curl \
    direnv \
    git \
    locales \
    lsb-release \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    ripgrep \
    sudo \
    tmux \
    vim \
    wget \
    zsh \
    && apt-get clean \
    && rm --recursive --force /var/lib/apt/lists/*

# TODO: Needed?
# Setup locale.
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen

# Install gcloud cli.
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install google-cloud-sdk -y \
    && apt-get clean \
    && rm --recursive --force /var/lib/apt/lists/*

# Install ngrok.
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
    && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | tee /etc/apt/sources.list.d/ngrok.list \
    && apt update \
    && apt install ngrok \
    && apt-get clean \
    && rm --recursive --force /var/lib/apt/lists/*

# Install docker.
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin \
    && apt-get clean \
    && rm --recursive --force /var/lib/apt/lists/*

# TODO: Update dependencies.
# Install python dev dependencies in separate environments via
# pipx to avoid dependency conflicts.
ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=/usr/local/py-utils/bin
ENV PATH=$PATH:$PIPX_BIN_DIR
RUN pip install pipx && pipx ensurepath
RUN pipx install black==22.3.0 \
    && pipx install dvc[gs]==2.10.1 \
    && pipx install flake8==4.0.1 \
    && pipx install isort==5.10.1 \
    && pipx install mypy==0.942 \
    && pipx install pre-commit==2.18.1 \
    && pipx install pytest==7.1.1

# Create non-root user.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN \
    groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/zsh $USERNAME \
    && usermod -G docker $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user.
USER $USERNAME

# Install Oh-my-zsh.
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
COPY custom.zsh-theme /home/$USERNAME/.oh-my-zsh/custom/themes/custom.zsh-theme
RUN sed -i -e 's/ZSH_THEME=.*/ZSH_THEME="custom"/g' ~/.zshrc
RUN sed -i -e 's/plugins.*/plugins=(git direnv)/g' ~/.zshrc

# Install fzf.
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install --all

# Append .local/bin to path.
ENV PATH=$PATH:/home/$USERNAME/.local/bin

# Install Pytorch dependencies for correct cuda version.
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install \
    diffusers==0.3.0 \
    ftfy \
    google-cloud-secret-manager \
    jupyter \
    numpy \
    pydantic \
    slack_bolt \
    transformers

ARG BASE_IMAGE_ARG=registry.access.redhat.com/ubi8:latest
FROM ${BASE_IMAGE_ARG}

COPY xvfb-rpms /tmp

RUN yum -y install python3.11 \
    python3.11-devel \
    python3.11-pip \
    mesa-libGL \
    python3.11-tkinter \
    xz \
    gcc \
    /tmp/libfontenc-1.1.3-8.el8.x86_64.rpm \
    /tmp/libICE-1.0.9-15.el8.x86_64.rpm \
    /tmp/libSM-1.2.3-1.el8.x86_64.rpm \
    /tmp/libXdmcp-1.1.3-1.el8.x86_64.rpm \
    /tmp/libXfont2-2.0.3-2.el8.x86_64.rpm \
    /tmp/libxkbfile-1.1.0-1.el8.x86_64.rpm \
    /tmp/libXmu-1.1.3-1.el8.x86_64.rpm \
    /tmp/libXt-1.1.5-12.el8.x86_64.rpm \
    /tmp/pixman-0.38.4-3.el8_9.x86_64.rpm \
    /tmp/python3-xvfbwrapper-0.2.9-2.el8.noarch.rpm \
    /tmp/xkeyboard-config-2.28-1.el8.noarch.rpm \
    /tmp/xorg-x11-server-common-1.20.11-17.el8.x86_64.rpm \
    /tmp/xorg-x11-server-Xvfb-1.20.11-17.el8.x86_64.rpm \
    /tmp/xorg-x11-xauth-1.0.9-12.el8.x86_64.rpm \
    /tmp/xorg-x11-xkb-utils-7.7-28.el8.x86_64.rpm


# Installing ffmpeg via relies on the rpmfusion repo and SDL2
# The SDL2 yum package is not currently available in ubi8
# Consequently, we download and install ffmpeg as a statically
# linked binary below.
WORKDIR /code
RUN curl -O https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
RUN tar -xf ffmpeg-*-amd64-static.tar.xz && \
    cp /code/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin && \
    cp /code/ffmpeg-*-amd64-static/ffprobe /usr/local/bin    

ENV PYTHONPATH=/code

COPY requirements.txt /code/
RUN python3 -m pip install -r requirements.txt && \
    python3 -m pip install pip-licenses && \
    pip-licenses
ENV PATH=$HOME/.local/bin:$PATH

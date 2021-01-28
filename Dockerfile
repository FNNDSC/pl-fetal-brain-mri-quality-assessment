FROM nvidia/cuda:11.1.1-base-ubuntu20.04 AS weights
ADD https://fnndsc.childrens.harvard.edu/mri_pipeline/ivan/quality_assessment/weights.tar.gz /tmp/weights.tar.gz
RUN ["tar", "xvf", "/tmp/weights.tar.gz", "--directory=/usr/local/src"]


FROM nvidia/cuda:11.1.1-base-ubuntu20.04 AS conda-installer
ENV CONDA_VERSION=4.8.3
ENV CONDA_PYTHON_VERSION=py38

RUN apt-get update \
    && apt-get install -qq curl \
    && curl -so /tmp/install-conda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_PYTHON_VERSION}_${CONDA_VERSION}-$(uname -s)-$(uname -p).sh \
    && bash /tmp/install-conda.sh -b -p /opt/conda


FROM nvidia/cuda:11.1.1-base-ubuntu20.04
WORKDIR /usr/local/src

COPY --from=weights /usr/local/src .

COPY --from=conda-installer /opt/conda /opt/conda
# better thing to do might be `conda init bash`
ENV PATH=/opt/conda/bin:$PATH


COPY conda_environment.yml .
RUN conda env update -n base -f conda_environment.yml

COPY . .
RUN pip install .


LABEL version="2.2.1" maintainer="FNNDSC <dev@babyMRI.org>"
CMD ["fetal_brain_quality_assessment", "--help"]

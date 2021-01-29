# - container image cuda version must match host cuda version
# - devel version is used instead of base because
#   because tensorflow makes calls to ptxas
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04 AS conda-installer
ENV CONDA_VERSION=4.8.3
ENV CONDA_PYTHON_VERSION=py38

RUN apt-get update \
    && apt-get install -qq curl \
    && curl -so /tmp/install-conda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_PYTHON_VERSION}_${CONDA_VERSION}-$(uname -s)-$(uname -p).sh \
    && bash /tmp/install-conda.sh -b -p /opt/conda


FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
WORKDIR /usr/local/src

COPY --from=conda-installer /opt/conda /opt/conda
# better thing to do might be `conda init bash`
ENV PATH=/opt/conda/bin:$PATH

COPY conda_environment.yml .
RUN conda env update -n base -f conda_environment.yml

# if you want to choose weights from a different fold, specify here.
# list of options at
# https://fnndsc.childrens.harvard.edu/mri_pipeline/ivan/quality_assessment/
ADD https://fnndsc.childrens.harvard.edu/mri_pipeline/ivan/quality_assessment/weights_resnet_sw2_k3.hdf5 \
    /usr/local/share/fetal_brain_quality_assessment/weights_resnet.hdf5
RUN ["chmod", "444", "/usr/local/share/fetal_brain_quality_assessment/weights_resnet.hdf5"]

COPY . .
RUN pip install .

CMD ["fetal_brain_quality_assessment", "--help"]

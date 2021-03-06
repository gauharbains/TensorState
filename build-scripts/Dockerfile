FROM quay.io/pypa/manylinux2014_x86_64

RUN /opt/python/cp36-cp36m/bin/pip install numpy cython==3.0a1 && \
    /opt/python/cp37-cp37m/bin/pip install numpy cython==3.0a1 && \
    /opt/python/cp38-cp38/bin/pip install numpy cython==3.0a1 && \
    /opt/python/cp39-cp39/bin/pip install numpy cython==3.0a1

RUN /opt/python/cp37-cp37m/bin/pip install twine

RUN git clone https://github.com/Nicholas-Schaub/TensorState.git

WORKDIR TensorState

RUN /opt/python/cp36-cp36m/bin/python setup.py sdist bdist_wheel && \
    /opt/python/cp37-cp37m/bin/python setup.py sdist bdist_wheel && \
    /opt/python/cp38-cp38/bin/python setup.py sdist bdist_wheel && \
    /opt/python/cp39-cp39/bin/python setup.py sdist bdist_wheel

RUN TS_VERSION=$(cat VERSION) &&\
    auditwheel repair ./dist/TensorState-${TS_VERSION}-cp36-cp36m-linux_x86_64.whl -w ./dist && \
    auditwheel repair ./dist/TensorState-${TS_VERSION}-cp37-cp37m-linux_x86_64.whl -w ./dist && \
    auditwheel repair ./dist/TensorState-${TS_VERSION}-cp38-cp38-linux_x86_64.whl -w ./dist && \
    auditwheel repair ./dist/TensorState-${TS_VERSION}-cp39-cp39-linux_x86_64.whl -w ./dist

ENTRYPOINT /opt/python/cp37-cp37m/bin/python -m twine upload --repository pypi dist/*manylinux2014* -u $TWINE_USER -p $TWINE_PASS

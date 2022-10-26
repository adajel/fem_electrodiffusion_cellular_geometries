# Base docker image
FROM ceciledc/fenics_mixed_dimensional:13-03-20

# Copy current directory
WORKDIR ${HOME}
COPY . ${HOME}

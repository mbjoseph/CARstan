FROM r-base
MAINTAINER Max Joseph maxwellbjoseph@gmail.com

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential libssl-dev pandoc

RUN install2.r --error \
    ggmcmc \
    gpclib \
    knitr \
    maptools \
    rmarkdown \
    rstan \
    spdep

COPY . /home/docker/CARstan
WORKDIR /home/docker/CARstan

CMD ["Rscript", "-e", "rmarkdown::render('README.Rmd')"]

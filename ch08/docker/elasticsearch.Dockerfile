FROM docker.elastic.co/elasticsearch/elasticsearch:7.14.0
RUN elasticsearch-plugin install analysis-nori 
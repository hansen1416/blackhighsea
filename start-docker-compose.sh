#!/bin/bash
docker-compose up -d
docker network connect --alias postgres blackhighsea_bhs postgres
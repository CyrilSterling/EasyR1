source .venv/bin/activate
set -x

which python

ssh_address=$(cat /etc/osmo_hosts.txt | tail -n 1)

ROOT_DIR=$(pwd)

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $ssh_address "source /etc/workflow_env && cd $ROOT_DIR && source .venv/bin/activate && bash EasyR1/scripts/deploy_judge/deploy_7b.sh"

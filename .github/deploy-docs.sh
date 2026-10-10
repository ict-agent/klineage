#!/usr/bin/env bash
set -euo pipefail

: "${DOCS_SSH_PRIVATE_KEY:?Set the DOCS_SSH_PRIVATE_KEY repository secret}"
release="${GITHUB_SHA:?}-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
[[ "$release" =~ ^[[:xdigit:]]{40}-[0-9]+-[0-9]+$ ]]
deploy_root=/var/www/klineage-docs
release_path="$deploy_root/releases/$release"

ssh_dir=$(mktemp -d)
trap 'rm -rf "$ssh_dir"' EXIT
printf '%s\n' "$DOCS_SSH_PRIVATE_KEY" > "$ssh_dir/key"
chmod 600 "$ssh_dir/key"
unset DOCS_SSH_PRIVATE_KEY

cat > "$ssh_dir/config" <<EOF
Host klineage-docs
    HostName 111.229.66.102
    User klineage-docs
    IdentityFile $ssh_dir/key
    UserKnownHostsFile $PWD/.github/docs_known_hosts
    StrictHostKeyChecking yes
    IdentitiesOnly yes
    BatchMode yes
    ConnectTimeout 15
EOF

# Upload a complete release before changing the live symlink.
ssh -F "$ssh_dir/config" klineage-docs "mkdir '$release_path'"
rsync -rtz --chmod=D755,F644 -e "ssh -F $ssh_dir/config" \
    site/ "klineage-docs:$release_path/"
ssh -F "$ssh_dir/config" klineage-docs bash -s -- "$deploy_root" "$release" <<'SH'
set -euo pipefail
deploy_root=$1
release=$2
test -s "$deploy_root/releases/$release/index.html"
test -s "$deploy_root/releases/$release/search/search_index.json"
ln -s "releases/$release" "$deploy_root/.current-$release"
mv -Tf "$deploy_root/.current-$release" "$deploy_root/current"
SH

curl --fail --silent --show-error --retry 3 \
    https://qiuchu.site/project/klineage/ --output "$ssh_dir/index.html"
cmp site/index.html "$ssh_dir/index.html"

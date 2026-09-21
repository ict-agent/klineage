# Source before launching Codex; never print credentials.
export KLINEAGE_PROVIDER_CONFIG=${KLINEAGE_PROVIDER_CONFIG:-$HOME/.config/klineage/deepseek.toml}
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  DEEPSEEK_KEY_FILE=${DEEPSEEK_KEY_FILE:-$HOME/.config/klineage/deepseek-api-key.txt}
  if [[ ! -r "$DEEPSEEK_KEY_FILE" ]]; then
    printf '%s\n' "Missing API key file: $DEEPSEEK_KEY_FILE" >&2
    return 1
  fi
  DEEPSEEK_API_KEY=$(tr -d '\r\n' < "$DEEPSEEK_KEY_FILE")
fi
if [[ -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
  printf '%s\n' 'Fill ~/.config/klineage/deepseek-api-key.txt with your DeepSeek API key first.' >&2
  return 1
fi
export DEEPSEEK_API_KEY

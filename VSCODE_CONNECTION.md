````markdown
# Koa cluster: connect VS Code derivatives (Antigravity / Cursor / Windsurf / VS Code Remote-SSH)

## One-time setup

### A) Add SSH config entries on your local machine

Edit your SSH config file and add the following entries.

- **Linux / macOS**: `~/.ssh/config`
- **Windows**: `C:\Users\<local-username>\.ssh\config` (Replace `<local-username>` with your Windows user name)

```sshconfig
Host koa
  HostName koa.its.hawaii.edu
  User <your-cluster-username>

# Dynamic target: forwards through koa to the first compute node in your SLURM queue
Host koa-compute
  HostName ignored
  User <your-cluster-username>
  ProxyCommand ssh koa "node=\$(squeue -h -u <your-cluster-username> -o %N | head -n 1); node=\$(echo \$node | cut -d, -f1); [ -n \"\$node\" ] || exit 1; exec ssh \$node -W localhost:22"
```

> [!NOTE]
> On **Windows**, you may need to use `%%N` instead of `%N` in the `ProxyCommand` if using the built-in OpenSSH client, as `%` is a special character.
> Replace `<your-cluster-username>` with your actual cluster username (e.g., `molybog`).

### B) Fix `noexec` home by moving the remote server dir to an executable filesystem

Cluster home directories are often mounted with `noexec`, which prevents IDE servers from running. Move the server directory to `/var/tmp` and symlink it.

1. SSH to the cluster login endpoint:
```bash
ssh koa
```

2. Create an exec-backed server directory in `/var/tmp` and symlink the expected path in `$HOME`:

Replace `.<editor>-server` with the directory specific to your IDE:
- **VS Code**: `.vscode-server`
- **Cursor**: `.cursor-server`
- **Windsurf**: `.windsurf-server`
- **Antigravity**: `.antigravity-server`

```bash
# Example for Antigravity (replace .antigravity-server if using a different IDE)
export EDITOR_SERVER=".antigravity-server" 
rm -rf ~/$EDITOR_SERVER
mkdir -p /var/tmp/$USER/$EDITOR_SERVER
chmod 700 /var/tmp/$USER /var/tmp/$USER/$EDITOR_SERVER
ln -s /var/tmp/$USER/$EDITOR_SERVER ~/$EDITOR_SERVER
ls -ld ~/$EDITOR_SERVER
```

---

## Every time you want to connect (per session)

### 1) Launch an interactive job (SLURM) on a compute node

From the login environment:

```bash
ssh koa
```

Request an interactive allocation:

```bash
# Replace <partition> and <time> as needed.
# Example: -p sandbox -t 04:00:00 (Max 4h for sandbox)
salloc -p <partition> -t <time>
srun --pty bash
hostname
```

> [!IMPORTANT]
> - **Sandbox Partition**: Maximum limit is **4 hours**.
> - **Other Partitions**: Time limits can be higher (see onboarding documentation).
> - Once the time limit is reached, your session will end and you will need to repeat this step to reconnect.

Keep this allocation alive while you work. `koa-compute` resolves to the node listed by your active job.

### 2) Connect your editor to the compute node

In your IDE (VS Code / Cursor / Windsurf / Antigravity):

* **Remote-SSH: Connect to Host...** -> **`koa-compute`**

This will:
* Jump through the `koa` login node.
* Land on your active compute node allocated by SLURM.

You will need to go through a two-factor authentication process, possibly answer a question (yes), and input your password once again, to connect to the cluster. 


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

### B) (Optional) Check for `noexec` issues

Most users can skip this step. The IDE server will install directly into your home directory (`~/.antigravity-server`), which is shared across all nodes. 

**Only** follow the "Troubleshooting" section below if you specifically encounter errors related to `noexec` or permission denied when starting the server.

---

## Every time you want to connect (per session)

### 1) Launch an interactive job (SLURM) on a compute node

From the login environment:

```bash
ssh koa
```

Hostname:
```bash
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

## Troubleshooting

### "Permission denied" or "Server failed to start" (noexec home)

If your home directory is mounted with `noexec`, the server installation will fail. You can workaround this by using `/var/tmp`, but **be accessible**: `/var/tmp` is node-local!

1. **On the login node (`koa`)**, create a symlink from `$HOME` to `/var/tmp`:
   ```bash
   ssh koa
   export EDITOR_SERVER=".antigravity-server" 
   rm -rf ~/$EDITOR_SERVER
   ln -s /var/tmp/$USER/$EDITOR_SERVER ~/$EDITOR_SERVER
   ```

2. **Every time you connect to a NEW compute node**, you must create the directory:
   ```bash
   # On the compute node
   mkdir -p /var/tmp/$USER/.antigravity-server && chmod 700 /var/tmp/$USER /var/tmp/$USER/.antigravity-server
   ```


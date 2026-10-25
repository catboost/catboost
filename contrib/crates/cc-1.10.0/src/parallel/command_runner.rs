use std::{
    cell::Cell,
    process::{Child, Command},
};

use crate::{
    cell_modify, cell_push,
    parallel::{
        async_executor::{block_on, YieldOnce},
        job_token,
        reactor::{ChildRegistration, Reactor, Registration},
    },
    spawn, CargoOutput, CommandLine, Error, ErrorKind, StderrForwarder,
};

struct KillOnDrop(Child, StderrForwarder);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let child = &mut self.0;

        child.kill().ok();
    }
}

fn try_wait_on_child(
    cmd: &Command,
    child: &mut Child,
    stderr_forwarder: &mut StderrForwarder,
    cargo_output: &CargoOutput,
) -> Result<Option<()>, Error> {
    stderr_forwarder.forward_available(cmd);

    match child.try_wait() {
        Ok(Some(status)) => {
            stderr_forwarder.forward_all(cmd);

            cargo_output.print_debug(&status);

            if status.success() {
                Ok(Some(()))
            } else {
                Err(cargo_output.command_failed(cmd, status))
            }
        }
        Ok(None) => Ok(None),
        Err(e) => {
            stderr_forwarder.forward_all(cmd);
            Err(Error::new(
                ErrorKind::ToolExecError,
                format!(
                    "failed to wait on spawned child process `{}`: {e}",
                    CommandLine(cmd)
                ),
            ))
        }
    }
}

pub(crate) fn run_commands_in_parallel(
    cargo_output: &CargoOutput,
    cmds: &mut dyn Iterator<Item = Result<Command, Error>>,
) -> Result<(), Error> {
    // When compiling objects in parallel we do a few dirty tricks to speed
    // things up:
    //
    // * First is that we use the `jobserver` crate to limit the parallelism
    //   of this build script. The `jobserver` crate will use a jobserver
    //   configured by Cargo for build scripts to ensure that parallelism is
    //   coordinated across C compilations and Rust compilations. Before we
    //   compile anything we make sure to wait until we acquire a token.
    //
    //   Note that this jobserver is cached globally so we only used one per
    //   process and only worry about creating it once.
    //
    // * Next we use spawn the process to actually compile objects in
    //   parallel after we've acquired a token to perform some work
    //
    // With all that in mind we compile all objects in a loop here, after we
    // acquire the appropriate tokens, Once all objects have been compiled
    // we wait on all the processes and propagate the results of compilation.

    let reactor = Reactor::default();

    // The registrations come before the child, so that they are dropped
    // first, while what they refer to is still open.
    let pendings = Cell::new(Vec::<(
        Command,
        Option<Registration<'_>>, // stderr_registration
        ChildRegistration<'_>,
        KillOnDrop,
        job_token::JobToken,
    )>::new());
    let is_disconnected = Cell::new(false);
    let has_made_progress = Cell::new(false);

    let wait_future = async {
        let mut error = None;

        loop {
            // If the other end of the pipe is already disconnected, then we're not gonna get any new jobs,
            // so it doesn't make sense to reuse the tokens; in fact,
            // releasing them as soon as possible (once we know that the other end is disconnected) is beneficial.
            // Imagine that the last file built takes an hour to finish; in this scenario,
            // by not releasing the tokens before that last file is done we would effectively block other processes from
            // starting sooner - even though we only need one token for that last file, not N others that were acquired.

            let pendings_is_empty = cell_modify(&pendings, |pendings| {
                // Try waiting on them.
                pendings.retain_mut(|(cmd, stderr_registration, _, child, _token)| {
                    match try_wait_on_child(cmd, &mut child.0, &mut child.1, cargo_output) {
                        Ok(Some(())) => {
                            // Task done, remove the entry
                            has_made_progress.set(true);
                            false
                        }
                        Ok(None) => {
                            // Task still not finished, keep the entry. Once
                            // its stderr is closed there is nothing more to
                            // wait for on it, and keeping it registered
                            // would wake every wait at once.
                            if child.1.stderr().is_none() {
                                *stderr_registration = None;
                            }
                            true
                        }
                        Err(err) => {
                            // Task fail, remove the entry.
                            // Since we can only return one error, log the error to make
                            // sure users always see all the compilation failures. The
                            // logger already got each failed command as `CommandFailed`.
                            has_made_progress.set(true);

                            if cargo_output.warnings {
                                println!("cargo:warning={err}");
                            }
                            error = Some(err);

                            false
                        }
                    }
                });
                pendings.is_empty()
            });

            if pendings_is_empty && is_disconnected.get() {
                break if let Some(err) = error {
                    Err(err)
                } else {
                    Ok(())
                };
            }

            YieldOnce::default().await;
        }
    };
    let spawn_future = async {
        // Limit our parallelism globally with a jobserver.
        let mut tokens = job_token::ActiveJobTokenServer::new();

        for res in cmds {
            let mut cmd = res?;
            let token = tokens.acquire(&reactor).await?;

            let mut child = spawn(&mut cmd, cargo_output)?;
            let stderr_forwarder = StderrForwarder::new(&mut child, cargo_output);
            let mut child = KillOnDrop(child, stderr_forwarder);

            child.1.set_non_blocking()?;
            let stderr_registration = child
                .1
                .stderr()
                .map(|stderr| reactor.register_stderr(stderr))
                .transpose()?;
            let child_registration = reactor.register_child(&child.0)?;

            cell_push(
                &pendings,
                (cmd, stderr_registration, child_registration, child, token),
            );

            has_made_progress.set(true);
        }
        is_disconnected.set(true);

        Ok::<_, Error>(())
    };

    block_on(wait_future, spawn_future, &has_made_progress, &reactor)
}

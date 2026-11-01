use std::{
    io,
    marker::PhantomData,
    process::{Child, ChildStderr},
    thread::sleep,
    time::Duration,
};

#[derive(Default)]
pub(crate) struct Reactor(PhantomData<()>);

pub(crate) struct Registration<'a>(PhantomData<&'a Reactor>);

pub(crate) struct ChildRegistration<'a>(PhantomData<&'a Reactor>);

impl Reactor {
    pub(crate) fn register_stderr(&self, _stderr: &ChildStderr) -> io::Result<Registration<'_>> {
        Ok(Registration(PhantomData))
    }

    pub(crate) fn register_child(&self, _child: &Child) -> io::Result<ChildRegistration<'_>> {
        Ok(ChildRegistration(PhantomData))
    }

    pub(crate) fn wait_with_timeout(&self, timeout: Duration) -> io::Result<()> {
        sleep(timeout);
        Ok(())
    }
}

export async function promisePool(tasks, poolLimit = 4) {
  const results = [];
  const executing = [];
  for (const task of tasks) {
    const p = task().then((r) => {
      executing.splice(executing.indexOf(p), 1);
      return r;
    });
    results.push(p);
    executing.push(p);
    if (executing.length >= poolLimit) {
      await Promise.race(executing);
    }
  }
  return Promise.all(results);
}

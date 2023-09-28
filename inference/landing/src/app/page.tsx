'use client'

import Image from "next/image";
import HeaderLink from "./components/headerLink";
import React from "react";
import {BenchmarkExplorer} from "./benchmark_explorer";

export default function Home() {
  const [page, setPage] = React.useState("landing");
  return (
    <main className="flex min-h-screen flex-col items-center justify-between gap-0">
      <div className="">Above the fold</div>
      <div className="flex flex-row items-center justify-between w-full">
        <button onClick={()=>setPage('snippets')}>snippets</button><button onClick={()=>setPage('benchmark-explorer')}>Benchmark Explorer</button>
      </div>
      {page === "snippets" && <div>Below the fold</div>}
      {page === "benchmark-explorer" && (
        <div style={{width:'100%'}}><BenchmarkExplorer /></div>
      )}
    </main>
  );
}

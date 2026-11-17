"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { sections } from "@/data/siteContent";
import { LiveStatusBadge } from "./LiveStatusBadge";

export function Navigation() {
  const [active, setActive] = useState("");

  useEffect(() => {
    const targets = sections.map(({ id }) => document.getElementById(id)).filter((target): target is HTMLElement => target !== null);
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setActive(visible.target.id);
    }, { rootMargin: "-20% 0px -60% 0px", threshold: [0, 0.25, 0.5] });
    targets.forEach((target) => observer.observe(target));
    return () => observer.disconnect();
  }, []);

  return <>
    <a className="skip-link" href="#top">Skip to content</a>
    <header className="site-nav">
      <div className="site-nav-left">
        <Link className="wordmark" href="/" aria-label="ROOMAI — Home">ROOMAI<span>®</span></Link>
      </div>
      <nav aria-label="Main navigation">
        {sections.map(({ id, number, label }) => (
          <a
            key={id}
            href={id === "explore" ? "#detect" : `#${id}`}
            aria-current={active === id ? "location" : undefined}
          >
            <span>{number}</span> {label}
          </a>
        ))}
      </nav>
      <div className="site-nav-right">
        <LiveStatusBadge showDetails={false} />
        <Link href="/model" className="nav-studio-btn">
          ● USE TRAINED MODEL ↗
        </Link>
      </div>
    </header>
  </>;
}

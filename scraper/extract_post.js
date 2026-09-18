// LinkedIn company-page post extractor.
//
// How the dataset was collected (2026-09-17):
//   1. Log into LinkedIn in a normal browser and open
//      https://www.linkedin.com/company/<slug>/posts/
//   2. Paste this whole file into the DevTools console once. It defines
//      window.__extractPosts().
//   3. Scroll with the mouse wheel / trackpad, clicking "Show more results" when it
//      appears, and call __extractPosts() every few scrolls. Results accumulate in
//      sessionStorage, so they survive LinkedIn's SPA navigations.
//   4. copy(__dumpPosts()) and save the output as data/raw/json/<page>.json
//      (wrap it as {"company", "company_followers", "scraped_at", "posts": [...]}).
//
// Why not a headless scraper: LinkedIn bounces automated sessions back to the feed
// within ~15 seconds of programmatic scrolling (we hit this repeatedly; real
// wheel events plus clicking the load button worked). Scraping also breaches
// LinkedIn's User Agreement, so this is kept to a small, manual, one-off
// research sample, not something to run at scale.
//
// Field names are short to keep dumps compact:
//   id  activity id (its top 41 bits are the post's creation time in ms)
//   t   first 300 chars of text        tl  full text length
//   h   hashtags   e  emoji   u  links   q  question marks   (all on the full text)
//   l   reactions  c  comments  r  reposts
//   m   media type  n  image count  ts  posted_at (UTC, from the activity id)
// Reposts of other accounts are skipped: their numbers belong to the original author.

(() => {
    const num = (s) => {
        if (!s) return 0;
        const m = s.replace(/,/g, '').match(/([\d.]+)\s*([KkMm]?)/);
        if (!m) return 0;
        let v = parseFloat(m[1]);
        if (/k/i.test(m[2])) v *= 1e3;
        if (/m/i.test(m[2])) v *= 1e6;
        return Math.round(v);
    };
    const grab = (t, re) => { const m = t.match(re); return m ? num(m[1]) : 0; };

    const mediaType = (p) => {
        const q = (s) => p.querySelector(s);
        if (q('.update-components-poll')) return ['poll', 0];
        if (q('.update-components-document__container, .document-s-container')) return ['document', 0];
        if (q('.update-components-linkedin-video, video')) return ['video', 0];
        if (q('.update-components-carousel, .feed-shared-carousel')) return ['carousel', 0];
        if (q('.update-components-image')) {
            const n = p.querySelectorAll('.update-components-image img').length;
            return [n > 1 ? 'multi_image' : 'image', n];
        }
        if (q('.update-components-article, .update-components-article-first-party, .update-components-entity')) return ['article', 0];
        return ['text_only', 0];
    };

    window.__extractPosts = () => {
        const slug = location.pathname.split('/')[2];
        const key = `__acc_${slug}`;
        const acc = JSON.parse(sessionStorage.getItem(key) || '{}');
        const f = (document.querySelector('main')?.innerText.match(/([\d,.]+K?M?)\s+followers/i) || [])[1];
        if (f) acc.__followers = f;

        for (const p of document.querySelectorAll('div[data-urn^="urn:li:activity"]')) {
            const q = (s) => p.querySelector(s);
            if (/reposted/i.test(q('.update-components-header')?.innerText || '')) continue;
            const text = (q('.update-components-text, .feed-shared-update-v2__description')?.innerText || '')
                .replace(/…more$/, '').replace(/hashtag\n#/g, '#').trim();
            const counts = (q('.social-details-social-counts')?.innerText || '').replace(/\n/g, ' ');
            const [m, n] = mediaType(p);
            const id = p.getAttribute('data-urn').split(':').pop();
            acc[id] = {
                id,
                t: text.slice(0, 300),
                tl: text.length,
                h: (text.match(/#\w+/g) || []).length,
                e: (text.match(/\p{Extended_Pictographic}/gu) || []).length,
                u: (text.match(/https?:\/\/|lnkd\.in/g) || []).length,
                q: (text.match(/\?/g) || []).length,
                l: num(q('.social-details-social-counts__reactions-count')?.innerText),
                c: grab(counts, /([\d,.]+[KkMm]?)\s+comments?/),
                r: grab(counts, /([\d,.]+[KkMm]?)\s+reposts?/),
                m,
                n,
                ts: new Date(Number(BigInt(id) >> 22n)).toISOString().slice(0, 16),
            };
        }
        sessionStorage.setItem(key, JSON.stringify(acc));
        return Object.keys(acc).filter((k) => !k.startsWith('__')).length;
    };

    window.__dumpPosts = () => {
        const slug = location.pathname.split('/')[2];
        const acc = JSON.parse(sessionStorage.getItem(`__acc_${slug}`) || '{}');
        return JSON.stringify(Object.values(acc).filter((p) => p && p.id), null, 0);
    };
})();

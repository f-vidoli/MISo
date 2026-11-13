/*
 * miso.c - conformalized mean curvature flow to a sphere with a Neo-Hookean
 *          tetrahedral interior.  C port with OpenMP and optional OpenCL.
 *
 * Per step
 *   1. Boundary target U_b:
 *        - cMCF   : (M + delta L) U_b = M V_old          (while NOT star-shaped)
 *        - radial : project every boundary vertex along its ray from a kernel
 *                   point c toward the sphere of target radius (once the
 *                   domain IS star-shaped w.r.t. c; no linear solve at all)
 *   2. All vertices: inexact Newton-CG on
 *        E(V) = E_NH(V) + 1/2 sum_b k_b |V_b - U_b|^2
 *      with step clipping before any tet reaches J=0 and Armijo backtracking.
 *   3. Recentre and rescale to constant area.
 *
 * Neo-Hookean:  Psi = mu/2 (I_C - 3) - mu ln J + kappa/2 (ln J)^2
 *
 * Build (CPU only):        cc -O3 -march=native -fopenmp miso.c -o miso -lm
 * Build (with OpenCL GPU): cc -O3 -march=native -fopenmp -DUSE_OPENCL miso.c -o miso -lOpenCL -lm
 *
 * Input:  .stl  -> surface-only flow (no interior)
 *         .tet  -> tetrahedral mesh ("nv nt", nv xyz lines, nt "a b c d" lines)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <ctype.h>
#include <signal.h>
#include <stdarg.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <unistd.h>
#define MKDIR(p) mkdir(p, 0777)
#endif
#ifdef USE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------------ */
/* small utilities                                                           */
/* ------------------------------------------------------------------------ */
static double now(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}
static void *xcalloc(size_t n, size_t s)
{
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n * s); exit(1); }
    return p;
}
static int u64cmp(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : (x > y);
}
static int find_in_row(const int *ci, int lo, int hi, int col)
{
    hi--;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (ci[mid] == col) return mid;
        if (ci[mid] < col) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}
static double dotd(const double *a, const double *b, int n)
{
    double s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static) if (n > 20000)
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static void spmv(int n, const int *rp, const int *ci, const double *va,
                 const double *x, double *y)
{
#pragma omp parallel for schedule(static) if (n > 4000)
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int k = rp[i]; k < rp[i + 1]; k++) s += va[k] * x[ci[k]];
        y[i] = s;
    }
}

/* ------------------------------------------------------------------------ */
/* terminal: pink-orange palette, icons, live progress rows                  */
/* ------------------------------------------------------------------------ */
#define ESC_R  "\033[0m"
#define C_PINK "\033[38;2;255;105;180m"   /* hot pink   */
#define C_SOFT "\033[2;38;2;255;175;200m" /* dim pink   */
#define C_ORNG "\033[38;2;255;153;51m"    /* orange     */
#define C_GOLD "\033[38;2;255;195;70m"    /* gold       */
#define C_WARN "\033[38;2;255;190;0m"
#define C_ERR  "\033[38;2;255;90;90m"
#define C_BOLD "\033[1m"
#define C_DIMW "\033[2;37m"

static int g_color, g_live;               /* ANSI colours / in-place live rows */
static char g_row1[1800], g_row2[900];
static int g_rows;                        /* live rows currently on screen     */

static void term_init(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);   /* keep stdout/stderr interleaved in logs */
#ifdef _WIN32
    g_color = g_live = _isatty(_fileno(stdout));
#else
    g_color = g_live = isatty(fileno(stdout));
#endif
    const char *nc = getenv("NO_COLOR");
    if (nc && *nc) g_color = 0;
}

static void hsl2rgb(double h, double s, double l, int rgb[3])
{
    h = fmod(h, 360.0); if (h < 0) h += 360;
    double c = (1.0 - fabs(2.0 * l - 1.0)) * s, x = c * (1.0 - fabs(fmod(h / 60.0, 2.0) - 1.0)), m = l - c / 2.0;
    double r = 0, g = 0, b = 0;
    if (h < 60) { r = c; g = x; } else if (h < 120) { r = x; g = c; }
    else if (h < 180) { g = c; b = x; } else if (h < 240) { g = x; b = c; }
    else if (h < 300) { r = x; b = c; } else { r = c; b = x; }
    rgb[0] = (int)((r + m) * 255.0 + 0.5); rgb[1] = (int)((g + m) * 255.0 + 0.5); rgb[2] = (int)((b + m) * 255.0 + 0.5);
}

static void p_line(FILE *out, const char *color, const char *icon, const char *fmt, ...)
{
    va_list ap;
    if (g_color) fputs(color, out);
    fprintf(out, "%s ", icon);
    va_start(ap, fmt); vfprintf(out, fmt, ap); va_end(ap);
    if (g_color) fputs(ESC_R, out);
    fputc('\n', out);
}
#define p_info(...) p_line(stdout, C_SOFT,  "\u2139", __VA_ARGS__)
#define p_ok(...)   p_line(stdout, C_ORNG,  "\u2714", __VA_ARGS__)
#define p_warn(...) p_line(stdout, C_WARN,  "\u26a0", __VA_ARGS__)
#define p_err(...)  p_line(stderr, C_ERR,   "\u2716", __VA_ARGS__)
#define p_hdr(...)  p_line(stdout, C_PINK,  "\u25b6", __VA_ARGS__)
#define p_step(...) p_line(stdout, C_GOLD,  "\u25b8", __VA_ARGS__)
#define p_star(...) p_line(stdout, C_BOLD C_PINK, "\u2728", __VA_ARGS__)

/* ---- two live rows (progress bar + stats), rewritten in place ------------ */
static void live_draw(void)
{
    if (!g_live || !g_row1[0]) return;
    if (g_rows > 0) printf("\033[%dA", g_rows);
    printf("\r\033[J%s\n%s\n", g_row1, g_row2);
    g_rows = 2;
    fflush(stdout);
}
static void live_set(const char *r1, const char *r2)
{
    if (!g_live) return;
    snprintf(g_row1, sizeof g_row1, "%s", r1);
    snprintf(g_row2, sizeof g_row2, "%s", r2);
    live_draw();
}
static void live_erase(void)
{
    if (g_rows > 0) { printf("\033[%dA\r\033[J", g_rows); g_rows = 0; fflush(stdout); }
}
static void live_end(void) { live_erase(); g_row1[0] = g_row2[0] = 0; }

static const char BAR_CH[4] = "\xe2\x94\x81";   /* U+2501 in UTF-8 */

/* line progress bar: pink->orange gradient over the filled part, dim tail */
static void bar_render(char *out, size_t n, double frac, int width)
{
    int fill = (int)(frac * width + 0.5);
    if (fill > width) fill = width;
    if (fill < 0) fill = 0;
    size_t o = 0;
    for (int k = 0; k < width && o + 32 < n; k++) {
        if (k < fill) {
            if (g_color) {
                int rgb[3];
                hsl2rgb(330.0 + 50.0 * k / (width > 1 ? width - 1 : 1), 0.95, 0.62, rgb);
                o += (size_t)snprintf(out + o, n - o, "\033[38;2;%d;%d;%dm\xe2\x94\x81", rgb[0], rgb[1], rgb[2]);
            } else { memcpy(out + o, BAR_CH, 3); o += 3; }
        } else {
            if (g_color) o += (size_t)snprintf(out + o, n - o, C_DIMW "\xe2\x94\x81");
            else { memcpy(out + o, BAR_CH, 3); o += 3; }
        }
    }
    snprintf(out + (o < n ? o : n - 1), n - (o < n ? o : n - 1), "%s", g_color ? ESC_R : "");
}
static void fmt_t(double sec, char *b, size_t n)
{
    int s = sec < 0 ? 0 : (int)(sec + 0.5);
    snprintf(b, n, "%d:%02d", s / 60, s % 60);
}
static const char *SPIN[10] = {"\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"};

/* ---- banner (art markers: \001 = pink, \002 = orange) -------------------- */
#define P1 "\001"
#define P2 "\002"
static const char *BANNER[] = {
    "                   /|",
    "▄─▄─▄ ─▄─ ▄─▄ ▄─▄ ( ゚ _ 。7",
    "▓ ▓ ▓  ▓  ▀─▄ ▓ ▓  |    \\ ノシ",
    "▀ ▀ ▀ ─▀─ ▀─▀ ▀─▀  ししと ) ",
};
#define NBANNER ((int)(sizeof BANNER / sizeof BANNER[0]))

static void banner_print(void)
{
    for (int i = 0; i < NBANNER; i++) {
        const char *s = BANNER[i];
        if (g_color) {
            int rgb[3];
            hsl2rgb(330.0 + 50.0 * i / (NBANNER > 1 ? NBANNER - 1 : 1), 0.9, 0.6, rgb);
            printf("\033[38;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
            for (; *s; s++) {
                if (*s == '\001') fputs(C_PINK, stdout);
                else if (*s == '\002') fputs(C_ORNG, stdout);
                else putchar(*s);
            }
            puts(ESC_R);
        } else {
            for (; *s; s++) if (*s != '\001' && *s != '\002') putchar(*s);
            putchar('\n');
        }
    }
    const char *sub = "The Morphoelastic Inverse problem SOlver";
    int n = (int)strlen(sub);
    for (int i = 0; sub[i]; i++) {
        if (g_color) {
            int rgb[3];
            hsl2rgb(330.0 + 50.0 * i / (n > 1 ? n - 1 : 1), 0.9, 0.6, rgb);
            printf("\033[38;2;%d;%d;%dm%c", rgb[0], rgb[1], rgb[2], sub[i]);
        } else putchar(sub[i]);
    }
    if (g_color) fputs(ESC_R, stdout);
    putchar('\n');
    putchar('\n');
}

typedef struct { double cmcf, star, asmb, prec, cg, ls, gpu_up; long cg_its; int gpu_solves, cpu_solves; } Timers;

/* ------------------------------------------------------------------------ */
/* options                                                                   */
/* ------------------------------------------------------------------------ */
typedef struct {
    const char *input, *out;
    int steps; double tol; int target_vol;
    double delta_scale, min_cot;
    double mu, kappa, fit; int ref_updated;
    int newton_iters; double newton_tol; int max_ls;
    double cg_tol; int cg_iters;
    int radial; double radial_rate; int star_every;
    const char *video, *python, *vsize, *script; int no_video, video_every, fps, xvfb;
    int shear, shear_steps; double shear_tol, fit_shear, shear_tang, kappa_shear;
    int gpu;              /* 0 off, 1 auto, 2 force */
    int gpu_min_rows, cl_plat, cl_dev, list_dev, threads;
    int allow_open;        /* proceed despite open / non-manifold / inconsistent STL */
} Args;

/* ------------------------------------------------------------------------ */
/* mesh                                                                      */
/* ------------------------------------------------------------------------ */
typedef struct {
    int nv, nt, nf, nb;
    double *V;
    int *T;      /* 4*nt or NULL */
    int *F;      /* 3*nf global ids, outward oriented */
    int *Fl;     /* 3*nf boundary-local ids */
    int *b;      /* nb global ids of boundary vertices (sorted) */
} Mesh;

static const int TET_FACES[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
static const int TET_OPP[4] = {3, 2, 1, 0};

typedef struct { float x, y, z; int id; } VKey;
static int vkcmp(const void *a, const void *b)
{
    const VKey *p = a, *q = b;
    if (p->x != q->x) return p->x < q->x ? -1 : 1;
    if (p->y != q->y) return p->y < q->y ? -1 : 1;
    if (p->z != q->z) return p->z < q->z ? -1 : 1;
    return 0;
}

static double signed_volume(const double *V, const int *F, int nf)
{
    double s = 0;
    for (int f = 0; f < nf; f++) {
        const double *a = V + 3 * F[3 * f], *b = V + 3 * F[3 * f + 1], *c = V + 3 * F[3 * f + 2];
        s += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
           + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    return s / 6.0;
}

/* undirected edge key for the STL safety scan: u < v, dir =0 as wound */
typedef struct { int u, v, dir; } HalfEdge;
static int hecmp(const void *a, const void *b)
{
    const HalfEdge *x = a, *y = b;
    if (x->u != y->u) return x->u - y->u;
    if (x->v != y->v) return x->v - y->v;
    return x->dir - y->dir;
}

/* STL (binary or ASCII).  Merges vertices, drops degenerate faces, flips all
 * faces if the enclosed volume is negative, then verifies the surface is
 * closed, 2-manifold and consistently wound (unless allow_open). */
static int load_stl(const char *path, Mesh *m, int allow_open)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { p_err("cannot open %s", path); return 0; }
    fseek(fp, 0, SEEK_END); long len = ftell(fp); fseek(fp, 0, SEEK_SET);
    unsigned char *buf = xmalloc((size_t)len + 1);
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) { fclose(fp); return 0; }
    buf[len] = 0; fclose(fp);

    int ntri = 0; float *tc = NULL;
    uint32_t nbin = 0;
    if (len >= 84) memcpy(&nbin, buf + 80, 4);
    if (len >= 84 && (long)(84 + 50 * (size_t)nbin) == len) {
        ntri = (int)nbin; tc = xmalloc(sizeof(float) * 9 * (size_t)ntri);
        for (int t = 0; t < ntri; t++)
            memcpy(tc + 9 * t, buf + 84 + 50 * (size_t)t + 12, 36);
    } else {
        int cap = 1024; tc = xmalloc(sizeof(float) * 9 * (size_t)cap);
        char *p = (char *)buf; int nvtx = 0;
        while ((p = strstr(p, "vertex")) != NULL) {
            p += 6;
            if (nvtx / 3 >= cap) { cap *= 2; tc = realloc(tc, sizeof(float) * 9 * (size_t)cap); }
            for (int k = 0; k < 3; k++) tc[3 * nvtx + k] = (float)strtod(p, &p);
            nvtx++;
        }
        ntri = nvtx / 3;
    }
    free(buf);
    if (ntri == 0) { p_err("no triangles in %s", path); return 0; }

    VKey *k = xmalloc(sizeof(VKey) * 3 * (size_t)ntri);
    for (int i = 0; i < 3 * ntri; i++) { k[i].x = tc[3 * i]; k[i].y = tc[3 * i + 1]; k[i].z = tc[3 * i + 2]; k[i].id = i; }
    qsort(k, 3 * (size_t)ntri, sizeof(VKey), vkcmp);
    int *vid = xmalloc(sizeof(int) * 3 * (size_t)ntri);
    int nv = 0;
    double *V = xmalloc(sizeof(double) * 9 * (size_t)ntri);
    for (int i = 0; i < 3 * ntri; i++) {
        if (i == 0 || vkcmp(&k[i], &k[i - 1]) != 0) {
            V[3 * nv] = k[i].x; V[3 * nv + 1] = k[i].y; V[3 * nv + 2] = k[i].z; nv++;
        }
        vid[k[i].id] = nv - 1;
    }
    int *F = xmalloc(sizeof(int) * 3 * (size_t)ntri); int nf = 0, ndeg = 0;
    for (int t = 0; t < ntri; t++) {
        int a = vid[3 * t], b = vid[3 * t + 1], c = vid[3 * t + 2];
        if (a == b || b == c || a == c) { ndeg++; continue; }
        F[3 * nf] = a; F[3 * nf + 1] = b; F[3 * nf + 2] = c; nf++;
    }
    free(k); free(vid); free(tc);
    if (nf == 0) { p_err("%s: no non-degenerate triangles", path); return 0; }
    double vol = signed_volume(V, F, nf);
    if (vol < 0)
        for (int f = 0; f < nf; f++) { int t = F[3 * f]; F[3 * f] = F[3 * f + 1]; F[3 * f + 1] = t; vol = -vol; }

    /* ---- safety: closed, 2-manifold, consistently wound ------------------ */
    HalfEdge *he = xmalloc(sizeof(HalfEdge) * 3 * (size_t)nf);
    for (int f = 0; f < nf; f++)
        for (int e = 0; e < 3; e++) {
            int a = F[3 * f + e], b = F[3 * f + (e + 1) % 3];
            HalfEdge *h = &he[3 * f + e];
            h->dir = a < b ? 0 : 1;
            h->u = a < b ? a : b;
            h->v = a < b ? b : a;
        }
    qsort(he, 3 * (size_t)nf, sizeof(HalfEdge), hecmp);
    int nbnd = 0, nnm = 0, nwind = 0;
    for (long i = 0; i < 3L * nf; ) {
        long j = i;
        while (j < 3L * nf && he[j].u == he[i].u && he[j].v == he[i].v) j++;
        int cnt = (int)(j - i);
        if (cnt == 1) nbnd++;
        else if (cnt > 2) nnm++;
        for (long e = i + 1; e < j; e++) if (he[e].dir == he[e - 1].dir) nwind++;
        i = j;
    }
    free(he);
    int bad = nbnd || nnm || nwind;
    if (bad && !allow_open) {
        p_err("%s: bad surface — %d open edge%s, %d non-manifold edge%s, %d winding conflict%s (rerun with --allow-open to proceed)",
              path, nbnd, nbnd == 1 ? "" : "s", nnm, nnm == 1 ? "" : "s", nwind, nwind == 1 ? "" : "s");
        return 0;
    }
    if (bad)
        p_warn("%s: open=%d non-manifold=%d winding=%d — proceeding (--allow-open)", path, nbnd, nnm, nwind);
    if (ndeg) p_warn("%s: %d degenerate triangle%s dropped", path, ndeg, ndeg == 1 ? "" : "s");
    if (!(vol > 0)) { p_err("%s: zero enclosed volume", path); return 0; }
    m->nv = nv; m->V = V; m->nf = nf; m->F = F; m->nt = 0; m->T = NULL;
    p_ok("%s: %d verts, %d tris — %s, volume %.6g", path, nv, nf,
         bad ? "inconsistent (forced)" : "watertight, oriented", vol);
    return 1;
}

static int load_tet(const char *path, Mesh *m)
{
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int nv, nt;
    if (fscanf(fp, "%d %d", &nv, &nt) != 2) { fclose(fp); return 0; }
    double *V = xmalloc(sizeof(double) * 3 * (size_t)nv);
    int *T = xmalloc(sizeof(int) * 4 * (size_t)nt);
    for (int i = 0; i < 3 * nv; i++)
        if (fscanf(fp, "%lf", &V[i]) != 1) { fclose(fp); return 0; }
    for (int i = 0; i < 4 * nt; i++)
        if (fscanf(fp, "%d", &T[i]) != 1) { fclose(fp); return 0; }
    fclose(fp);
    int flipped = 0;                    /* enforce positive orientation */
    for (int t = 0; t < nt; t++) {
        const double *p0 = V + 3 * T[4 * t];
        double e[3][3];
        for (int k = 0; k < 3; k++) for (int i = 0; i < 3; i++) e[k][i] = V[3 * T[4 * t + k + 1] + i] - p0[i];
        double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
                   - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
                   + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
        if (det < 0) { int s = T[4 * t + 2]; T[4 * t + 2] = T[4 * t + 3]; T[4 * t + 3] = s; flipped++; }
    }
    if (flipped) p_info("reoriented %d tets to positive volume", flipped);
    m->nv = nv; m->nt = nt; m->V = V; m->T = T;
    return 1;
}

typedef struct { int s[3], o[3], opp; } BFace;
static int bfcmp(const void *a, const void *b)
{
    const BFace *p = a, *q = b;
    for (int k = 0; k < 3; k++) if (p->s[k] != q->s[k]) return p->s[k] < q->s[k] ? -1 : 1;
    return 0;
}
static void boundary_from_tets(Mesh *m)
{
    size_t n = 4 * (size_t)m->nt;
    BFace *bf = xmalloc(sizeof(BFace) * n);
    for (int t = 0; t < m->nt; t++)
        for (int f = 0; f < 4; f++) {
            BFace *e = &bf[4 * (size_t)t + f];
            for (int k = 0; k < 3; k++) e->o[k] = m->T[4 * t + TET_FACES[f][k]];
            e->opp = m->T[4 * t + TET_OPP[f]];
            memcpy(e->s, e->o, sizeof(e->s));
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2 - i; j++)
                if (e->s[j] > e->s[j + 1]) { int s = e->s[j]; e->s[j] = e->s[j + 1]; e->s[j + 1] = s; }
        }
    qsort(bf, n, sizeof(BFace), bfcmp);
    int *F = xmalloc(sizeof(int) * 3 * n); int nf = 0;
    for (size_t i = 0; i < n;) {
        size_t j = i + 1;
        while (j < n && bfcmp(&bf[i], &bf[j]) == 0) j++;
        if (j - i == 1) {
            int *o = bf[i].o;
            const double *a = m->V + 3 * o[0], *b = m->V + 3 * o[1], *c = m->V + 3 * o[2], *p = m->V + 3 * bf[i].opp;
            double u[3], v[3], nn[3], d = 0;
            for (int k = 0; k < 3; k++) { u[k] = b[k] - a[k]; v[k] = c[k] - a[k]; }
            nn[0] = u[1] * v[2] - u[2] * v[1]; nn[1] = u[2] * v[0] - u[0] * v[2]; nn[2] = u[0] * v[1] - u[1] * v[0];
            for (int k = 0; k < 3; k++) d += nn[k] * (p[k] - (a[k] + b[k] + c[k]) / 3.0);
            if (d > 0) { int s = o[0]; o[0] = o[1]; o[1] = s; }
            F[3 * nf] = o[0]; F[3 * nf + 1] = o[1]; F[3 * nf + 2] = o[2]; nf++;
        }
        i = j;
    }
    free(bf);
    m->F = F; m->nf = nf;
}

static void build_boundary_index(Mesh *m)
{
    int *map = xcalloc((size_t)m->nv, sizeof(int));
    for (int i = 0; i < 3 * m->nf; i++) map[m->F[i]] = 1;
    int nb = 0;
    for (int i = 0; i < m->nv; i++) if (map[i]) map[i] = nb++; else map[i] = -1;
    m->b = xmalloc(sizeof(int) * (size_t)nb);
    for (int i = 0; i < m->nv; i++) if (map[i] >= 0) m->b[map[i]] = i;
    m->Fl = xmalloc(sizeof(int) * 3 * (size_t)m->nf);
    for (int i = 0; i < 3 * m->nf; i++) m->Fl[i] = map[m->F[i]];
    m->nb = nb;
    free(map);
}

/* ------------------------------------------------------------------------ */
/* output                                                                    */
/* ------------------------------------------------------------------------ */
static void write_vtk(const Mesh *m, const double *V, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# vtk DataFile Version 3.0\nmiso\nASCII\n");
    fprintf(f, m->T ? "DATASET UNSTRUCTURED_GRID\n" : "DATASET POLYDATA\n");
    fprintf(f, "POINTS %d double\n", m->nv);
    for (int i = 0; i < m->nv; i++) fprintf(f, "%.10g %.10g %.10g\n", V[3 * i], V[3 * i + 1], V[3 * i + 2]);
    if (m->T) {
        fprintf(f, "CELLS %d %d\n", m->nt, 5 * m->nt);
        for (int t = 0; t < m->nt; t++) fprintf(f, "4 %d %d %d %d\n", m->T[4 * t], m->T[4 * t + 1], m->T[4 * t + 2], m->T[4 * t + 3]);
        fprintf(f, "CELL_TYPES %d\n", m->nt);
        for (int t = 0; t < m->nt; t++) fprintf(f, "10\n");
    } else {
        fprintf(f, "POLYGONS %d %d\n", m->nf, 4 * m->nf);
        for (int i = 0; i < m->nf; i++) fprintf(f, "3 %d %d %d\n", m->F[3 * i], m->F[3 * i + 1], m->F[3 * i + 2]);
    }
    fclose(f);
}
static void write_tet(const Mesh *m, const double *V, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%d %d\n", m->nv, m->nt);
    for (int i = 0; i < m->nv; i++) fprintf(f, "%.17g %.17g %.17g\n", V[3 * i], V[3 * i + 1], V[3 * i + 2]);
    for (int t = 0; t < m->nt; t++) fprintf(f, "%d %d %d %d\n", m->T[4 * t], m->T[4 * t + 1], m->T[4 * t + 2], m->T[4 * t + 3]);
    fclose(f);
}
static void write_stl(const Mesh *m, const double *V, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return;
    char hdr[80] = "miso boundary";
    fwrite(hdr, 1, 80, f);
    uint32_t n = (uint32_t)m->nf; fwrite(&n, 4, 1, f);
    for (int i = 0; i < m->nf; i++) {
        const double *a = V + 3 * m->F[3 * i], *b = V + 3 * m->F[3 * i + 1], *c = V + 3 * m->F[3 * i + 2];
        double u[3], v[3], nn[3];
        for (int k = 0; k < 3; k++) { u[k] = b[k] - a[k]; v[k] = c[k] - a[k]; }
        nn[0] = u[1] * v[2] - u[2] * v[1]; nn[1] = u[2] * v[0] - u[0] * v[2]; nn[2] = u[0] * v[1] - u[1] * v[0];
        double l = sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]); if (l < 1e-300) l = 1;
        float rec[12] = {(float)(nn[0] / l), (float)(nn[1] / l), (float)(nn[2] / l),
                         (float)a[0], (float)a[1], (float)a[2], (float)b[0], (float)b[1], (float)b[2],
                         (float)c[0], (float)c[1], (float)c[2]};
        uint16_t attr = 0;
        fwrite(rec, 4, 12, f); fwrite(&attr, 2, 1, f);
    }
    fclose(f);
}
/* ------------------------------------------------------------------------ */
/* surface geometry (boundary-local indexing)                               */
/* ------------------------------------------------------------------------ */
static void gather(const double *V, const int *b, int nb, double *S)
{
    for (int i = 0; i < nb; i++) memcpy(S + 3 * i, V + 3 * b[i], 3 * sizeof(double));
}
static double surf_area(const double *S, const int *F, int nf)
{
    double A = 0;
    for (int f = 0; f < nf; f++) {
        const double *a = S + 3 * F[3 * f], *b = S + 3 * F[3 * f + 1], *c = S + 3 * F[3 * f + 2];
        double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        double x = u[1] * v[2] - u[2] * v[1], y = u[2] * v[0] - u[0] * v[2], z = u[0] * v[1] - u[1] * v[0];
        A += 0.5 * sqrt(x * x + y * y + z * z);
    }
    return A;
}
static double mean_edge(const double *S, const int *F, int nf)
{
    double s = 0;
    for (int f = 0; f < nf; f++)
        for (int k = 0; k < 3; k++) {
            const double *a = S + 3 * F[3 * f + k], *b = S + 3 * F[3 * f + (k + 1) % 3];
            s += sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
        }
    return s / (3.0 * nf);
}
static void lumped_mass(const double *S, const int *F, int nf, int nb, double *m)
{
    memset(m, 0, sizeof(double) * (size_t)nb);
    for (int f = 0; f < nf; f++) {
        const double *a = S + 3 * F[3 * f], *b = S + 3 * F[3 * f + 1], *c = S + 3 * F[3 * f + 2];
        double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        double x = u[1] * v[2] - u[2] * v[1], y = u[2] * v[0] - u[0] * v[2], z = u[0] * v[1] - u[1] * v[0];
        double ar = sqrt(x * x + y * y + z * z) / 6.0;
        for (int k = 0; k < 3; k++) m[F[3 * f + k]] += ar;
    }
}

/* ---- cMCF operator ------------------------------------------------------ */
typedef struct { int n, nf; int *rp, *ci, *epos, *dpos; double *va; } SurfOp;

static void surf_init(SurfOp *s, const int *F, int nf, int n)
{
    s->n = n; s->nf = nf;
    size_t cnt = 6 * (size_t)nf + (size_t)n, c = 0;
    uint64_t *keys = xmalloc(sizeof(uint64_t) * cnt);
    for (int f = 0; f < nf; f++)
        for (int k = 0; k < 3; k++) {
            uint64_t p = (uint64_t)F[3 * f + k], q = (uint64_t)F[3 * f + (k + 1) % 3];
            keys[c++] = p * n + q; keys[c++] = q * n + p;
        }
    for (int i = 0; i < n; i++) keys[c++] = (uint64_t)i * n + i;
    qsort(keys, cnt, sizeof(uint64_t), u64cmp);
    size_t m = 0;
    for (size_t i = 0; i < cnt; i++) if (i == 0 || keys[i] != keys[i - 1]) keys[m++] = keys[i];
    s->rp = xcalloc((size_t)n + 1, sizeof(int)); s->ci = xmalloc(sizeof(int) * m);
    for (size_t i = 0; i < m; i++) { s->rp[keys[i] / n + 1]++; s->ci[i] = (int)(keys[i] % n); }
    for (int i = 0; i < n; i++) s->rp[i + 1] += s->rp[i];
    free(keys);
    s->va = xmalloc(sizeof(double) * m);
    s->epos = xmalloc(sizeof(int) * 6 * (size_t)nf);
    s->dpos = xmalloc(sizeof(int) * (size_t)n);
    for (int f = 0; f < nf; f++)
        for (int k = 0; k < 3; k++) {
            int p = F[3 * f + (k + 1) % 3], q = F[3 * f + (k + 2) % 3];
            s->epos[6 * f + 2 * k] = find_in_row(s->ci, s->rp[p], s->rp[p + 1], q);
            s->epos[6 * f + 2 * k + 1] = find_in_row(s->ci, s->rp[q], s->rp[q + 1], p);
        }
    for (int i = 0; i < n; i++) s->dpos[i] = find_in_row(s->ci, s->rp[i], s->rp[i + 1], i);
}

/* Jacobi-preconditioned CG for SPD systems (double). */
static int pcg_jacobi(int n, const int *rp, const int *ci, const double *va, const double *dg,
                      const double *b, double *x, double tol, int maxit)
{
    double *r = xmalloc(sizeof(double) * 4 * (size_t)n), *z = r + n, *p = z + n, *Ap = p + n;
    memset(x, 0, sizeof(double) * (size_t)n);
    memcpy(r, b, sizeof(double) * (size_t)n);
    double bn = sqrt(dotd(b, b, n)); int k = 0;
    if (bn == 0) { free(r); return 0; }
    for (int i = 0; i < n; i++) { z[i] = r[i] / dg[i]; p[i] = z[i]; }
    double rz = dotd(r, z, n);
    for (; k < maxit; k++) {
        spmv(n, rp, ci, va, p, Ap);
        double pAp = dotd(p, Ap, n);
        if (!(pAp > 0)) break;
        double al = rz / pAp;
        for (int i = 0; i < n; i++) { x[i] += al * p[i]; r[i] -= al * Ap[i]; }
        if (sqrt(dotd(r, r, n)) <= tol * bn) { k++; break; }
        for (int i = 0; i < n; i++) z[i] = r[i] / dg[i];
        double rz2 = dotd(r, z, n), be = rz2 / rz; rz = rz2;
        for (int i = 0; i < n; i++) p[i] = z[i] + be * p[i];
    }
    free(r);
    return k;
}

/* One implicit cMCF step: (M + delta L) U = M S, solved for the increment. */
static int cmcf_step(SurfOp *s, const double *S, const int *F, double delta, double min_cot,
                     const double *mass, double *U)
{
    int n = s->n, nf = s->nf;
    size_t nnz = (size_t)s->rp[n];
    memset(s->va, 0, sizeof(double) * nnz);
    double *dg = xcalloc((size_t)n, sizeof(double)), *Lv = xcalloc(3 * (size_t)n, sizeof(double));
    double mmax = 0; for (int i = 0; i < n; i++) if (mass[i] > mmax) mmax = mass[i];
    for (int f = 0; f < nf; f++)
        for (int k = 0; k < 3; k++) {
            int i = F[3 * f + k], p = F[3 * f + (k + 1) % 3], q = F[3 * f + (k + 2) % 3];
            const double *Si = S + 3 * i, *Sp = S + 3 * p, *Sq = S + 3 * q;
            double u[3] = {Sp[0] - Si[0], Sp[1] - Si[1], Sp[2] - Si[2]};
            double v[3] = {Sq[0] - Si[0], Sq[1] - Si[1], Sq[2] - Si[2]};
            double cs = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            double cx = u[1] * v[2] - u[2] * v[1], cy = u[2] * v[0] - u[0] * v[2], cz = u[0] * v[1] - u[1] * v[0];
            double sn = sqrt(cx * cx + cy * cy + cz * cz); if (sn < 1e-30) sn = 1e-30;
            double cot = cs / sn; if (cot < min_cot) cot = min_cot;
            double w = 0.5 * cot;
            s->va[s->epos[6 * f + 2 * k]] -= delta * w;
            s->va[s->epos[6 * f + 2 * k + 1]] -= delta * w;
            dg[p] += delta * w; dg[q] += delta * w;
            for (int c = 0; c < 3; c++) { double d = Sp[c] - Sq[c]; Lv[3 * p + c] += w * d; Lv[3 * q + c] -= w * d; }
        }
    for (int i = 0; i < n; i++) { dg[i] += mass[i] + 1e-12 * mmax; s->va[s->dpos[i]] = dg[i]; }
    double *rhs = xmalloc(sizeof(double) * 2 * (size_t)n), *x = rhs + n;
    int ok = 1;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < n; i++) rhs[i] = -delta * Lv[3 * i + c];
        pcg_jacobi(n, s->rp, s->ci, s->va, dg, rhs, x, 1e-10, 2000);
        for (int i = 0; i < n; i++) { U[3 * i + c] = S[3 * i + c] + x[i]; if (!isfinite(x[i])) ok = 0; }
    }
    free(dg); free(Lv); free(rhs);
    return ok;
}

/* ---- star-shapedness and radial projection ----------------------------- */
/* min over faces of the signed distance from c to the face plane; >0 means
 * every face is front-facing from c, i.e. the closed surface is star-shaped. */
static double star_margin(const double *S, const int *F, int nf, const double *c)
{
    double mn = 1e300;
    for (int f = 0; f < nf; f++) {
        const double *a = S + 3 * F[3 * f], *b = S + 3 * F[3 * f + 1], *d = S + 3 * F[3 * f + 2];
        double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
        double x = u[1] * v[2] - u[2] * v[1], y = u[2] * v[0] - u[0] * v[2], z = u[0] * v[1] - u[1] * v[0];
        double l = sqrt(x * x + y * y + z * z);
        if (l < 1e-300) continue;
        double s = (x * (a[0] - c[0]) + y * (a[1] - c[1]) + z * (a[2] - c[2])) / l;
        if (s < mn) mn = s;
    }
    return mn;
}
/* Alternating projections onto the half-spaces {x : n_f.(p_f - x) >= margin}.
 * Converges to a kernel point if the kernel is non-empty. */
static int find_kernel_point(const double *S, const int *F, int nf, int nb, double h,
                             double *c, int sweeps, double thr)
{
    double best = star_margin(S, F, nf, c);
    if (best > thr * h) return 1;
    double mean[3] = {0, 0, 0};
    for (int i = 0; i < nb; i++) for (int k = 0; k < 3; k++) mean[k] += S[3 * i + k] / nb;
    if (star_margin(S, F, nf, mean) > best) memcpy(c, mean, sizeof mean);
    static const double margins[3] = {0.1, 0.02, 0.002};
    for (int mi = 0; mi < 3; mi++) {
        double mg = margins[mi] * h;
        for (int sw = 0; sw < sweeps; sw++) {
            int changed = 0;
            for (int f = 0; f < nf; f++) {
                const double *a = S + 3 * F[3 * f], *b = S + 3 * F[3 * f + 1], *d = S + 3 * F[3 * f + 2];
                double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
                double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
                double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (l < 1e-300) continue;
                double s = (n[0] * (a[0] - c[0]) + n[1] * (a[1] - c[1]) + n[2] * (a[2] - c[2])) / l;
                if (s < mg) { double k = (mg - s) / l; for (int q = 0; q < 3; q++) c[q] -= k * n[q]; changed = 1; }
            }
            if (!changed) break;
        }
        if (star_margin(S, F, nf, c) > thr * h) return 1;
    }
    return 0;
}
/* r <- r^(1-rate) R^rate along the ray from c. */
static void radial_target(const double *S, int nb, const double *c, double R, double rate, double *U)
{
    for (int i = 0; i < nb; i++) {
        double d[3] = {S[3 * i] - c[0], S[3 * i + 1] - c[1], S[3 * i + 2] - c[2]};
        double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (r < 1e-12 * R) { memcpy(U + 3 * i, S + 3 * i, 3 * sizeof(double)); continue; }
        double rn = exp((1.0 - rate) * log(r) + rate * log(R)), k = rn / r;
        for (int q = 0; q < 3; q++) U[3 * i + q] = c[q] + d[q] * k;
    }
}

/* ------------------------------------------------------------------------ */
/* Neo-Hookean tet elasticity                                                */
/* ------------------------------------------------------------------------ */
typedef struct {
    int nv, nt, n, nnz, mode; const int *T; double mu, kappa;   /* mode 0: Neo-Hookean, 1: off-diagonal Cauchy shear + (ln J)^2 */
    double *vol, *gs;          /* gs: nt*12, F_ij = sum_a x_a,i gs[a][j] */
    int *bp, *bc, nnzb;        /* block CSR */
    int *bpos, *kd;            /* bpos: nt*16 block index for (a,b); kd: diag block */
    int *srp, *sci;            /* scalar CSR (3nv rows) */
    double *va;
} NH;

static inline double det3(const double *m)
{
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}
static inline double inv3(const double *m, double *o)
{
    double c00 = m[4] * m[8] - m[5] * m[7], c01 = m[5] * m[6] - m[3] * m[8], c02 = m[3] * m[7] - m[4] * m[6];
    double det = m[0] * c00 + m[1] * c01 + m[2] * c02, id = 1.0 / det;
    o[0] = c00 * id; o[1] = (m[2] * m[7] - m[1] * m[8]) * id; o[2] = (m[1] * m[5] - m[2] * m[4]) * id;
    o[3] = c01 * id; o[4] = (m[0] * m[8] - m[2] * m[6]) * id; o[5] = (m[2] * m[3] - m[0] * m[5]) * id;
    o[6] = c02 * id; o[7] = (m[1] * m[6] - m[0] * m[7]) * id; o[8] = (m[0] * m[4] - m[1] * m[3]) * id;
    return det;
}
static inline void defgrad(const double *X, const int *t4, const double *g, double *F)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int a = 0; a < 4; a++) s += X[3 * t4[a] + i] * g[3 * a + j];
            F[3 * i + j] = s;
        }
}

static void nh_init(NH *h, const int *T, int nt, int nv, double mu, double kappa)
{
    memset(h, 0, sizeof *h);
    h->nv = nv; h->nt = nt; h->T = T; h->mu = mu; h->kappa = kappa; h->n = 3 * nv;
    h->vol = xmalloc(sizeof(double) * (size_t)nt); h->gs = xmalloc(sizeof(double) * 12 * (size_t)nt);

    size_t cnt = 16 * (size_t)nt + (size_t)nv, c = 0;
    uint64_t *keys = xmalloc(sizeof(uint64_t) * cnt);
    for (int t = 0; t < nt; t++)
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++)
            keys[c++] = (uint64_t)T[4 * t + a] * nv + (uint64_t)T[4 * t + b];
    for (int i = 0; i < nv; i++) keys[c++] = (uint64_t)i * nv + i;
    qsort(keys, cnt, sizeof(uint64_t), u64cmp);
    size_t m = 0;
    for (size_t i = 0; i < cnt; i++) if (i == 0 || keys[i] != keys[i - 1]) keys[m++] = keys[i];
    h->nnzb = (int)m;
    h->bp = xcalloc((size_t)nv + 1, sizeof(int)); h->bc = xmalloc(sizeof(int) * m);
    for (size_t i = 0; i < m; i++) { h->bp[keys[i] / nv + 1]++; h->bc[i] = (int)(keys[i] % nv); }
    for (int i = 0; i < nv; i++) h->bp[i + 1] += h->bp[i];
    free(keys);

    h->bpos = xmalloc(sizeof(int) * 16 * (size_t)nt);
    for (int t = 0; t < nt; t++)
        for (int a = 0; a < 4; a++) {
            int i = T[4 * t + a];
            for (int b = 0; b < 4; b++)
                h->bpos[16 * t + 4 * a + b] = find_in_row(h->bc, h->bp[i], h->bp[i + 1], T[4 * t + b]);
        }
    h->kd = xmalloc(sizeof(int) * (size_t)nv);
    for (int i = 0; i < nv; i++) h->kd[i] = find_in_row(h->bc, h->bp[i], h->bp[i + 1], i);

    h->n = 3 * nv; h->nnz = 9 * h->nnzb;
    h->srp = xmalloc(sizeof(int) * ((size_t)h->n + 1)); h->sci = xmalloc(sizeof(int) * (size_t)h->nnz);
    for (int i = 0; i < nv; i++) {
        int nb = h->bp[i + 1] - h->bp[i];
        for (int r = 0; r < 3; r++) {
            int row = 3 * i + r, st = 9 * h->bp[i] + r * 3 * nb;
            h->srp[row] = st;
            for (int k = 0; k < nb; k++) for (int cc = 0; cc < 3; cc++) h->sci[st + 3 * k + cc] = 3 * h->bc[h->bp[i] + k] + cc;
        }
    }
    h->srp[h->n] = h->nnz;
    h->va = xmalloc(sizeof(double) * (size_t)h->nnz);
}

static void nh_set_ref(NH *h, const double *Vr)
{
    for (int t = 0; t < h->nt; t++) {
        const int *T4 = h->T + 4 * t;
        double D[9], Di[9];
        for (int k = 0; k < 3; k++) for (int i = 0; i < 3; i++) D[3 * i + k] = Vr[3 * T4[k + 1] + i] - Vr[3 * T4[0] + i];
        double det = inv3(D, Di);
        h->vol[t] = fabs(det) / 6.0;
        double *g = h->gs + 12 * t;
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) { g[3 * (k + 1) + j] = Di[3 * k + j]; s += Di[3 * k + j]; }
            g[j] = -s;
        }
    }
}

/* ---- Cauchy off-diagonal (shear) measure -----------------------------------
 * For Neo-Hookean, sigma = (mu (F F^T - I) + kappa ln J I)/J, so the off-diagonal
 * Cauchy stress is mu b_ij / J (b = F F^T): the volumetric part never enters.
 * We minimise  Phi = (1/mu) int_current sum_{i<j} sigma_ij^2 dV
 *                  = mu sum_t vol_t Q / J,   Q = b01^2 + b02^2 + b12^2.            */
static inline double cauchy_Q(const double *F, double *Boff)
{
    double b01 = F[0] * F[3] + F[1] * F[4] + F[2] * F[5], b02 = F[0] * F[6] + F[1] * F[7] + F[2] * F[8],
           b12 = F[3] * F[6] + F[4] * F[7] + F[5] * F[8];
    Boff[0] = 0; Boff[1] = b01; Boff[2] = b02; Boff[3] = b01; Boff[4] = 0; Boff[5] = b12;
    Boff[6] = b02; Boff[7] = b12; Boff[8] = 0;
    return b01 * b01 + b02 * b02 + b12 * b12;
}
/* first Piola stress of Phi/vol (without kappa term) */
static inline void cauchy_P(const double *F, const double *Ft, double mu, double J, double *P)
{
    double Bo[9], Q = cauchy_Q(F, Bo);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int m = 0; m < 3; m++) s += Bo[3 * i + m] * F[3 * m + j];
            P[3 * i + j] = mu * (2.0 * s - Q * Ft[3 * i + j]) / J;
        }
}
/* exact 9x9 Hessian d2(Phi/vol)/dF dF, index (3*i+J, 3*k+L); Fi = F^-1 */
static inline void cauchy_H(const double *F, const double *Fi, double mu, double J, double *H)
{
    double Bo[9], Ft[9], dQ[9], db[3][9];
    double Q = cauchy_Q(F, Bo);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Ft[3 * i + j] = Fi[3 * j + i];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int m = 0; m < 3; m++) s += Bo[3 * i + m] * F[3 * m + j];
            dQ[3 * i + j] = 2.0 * s;
        }
    static const int PQ[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int n = 0; n < 3; n++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                db[n][3 * i + j] = (PQ[n][0] == i ? F[3 * PQ[n][1] + j] : 0.0) + (PQ[n][1] == i ? F[3 * PQ[n][0] + j] : 0.0);
    for (int i = 0; i < 3; i++)
        for (int J2 = 0; J2 < 3; J2++)
            for (int k = 0; k < 3; k++)
                for (int L = 0; L < 3; L++) {
                    int r = 3 * i + J2, c = 3 * k + L;
                    double d1 = db[0][r] * db[0][c] + db[1][r] * db[1][c] + db[2][r] * db[2][c];
                    double v = 2.0 * d1 + (J2 == L ? 2.0 * Bo[3 * i + k] : 0.0) - dQ[r] * Ft[c] - Ft[r] * dQ[c]
                             + Q * Ft[r] * Ft[c] + Q * Fi[3 * J2 + k] * Fi[3 * L + i];
                    H[9 * r + c] = mu * v / J;
                }
}

static double nh_energy(const NH *h, const double *X)
{
    double e = 0; int bad = 0;
#pragma omp parallel for reduction(+ : e, bad) schedule(static)
    for (int t = 0; t < h->nt; t++) {
        double F[9]; defgrad(X, h->T + 4 * t, h->gs + 12 * t, F);
        double J = det3(F);
        if (!(J > 0)) { bad++; continue; }
        double lj = log(J), f2 = 0;
        if (h->mode == 1) {
            double Bo[9], Q = cauchy_Q(F, Bo);
            e += h->vol[t] * (h->mu * Q / J + 0.5 * h->kappa * lj * lj);
            continue;
        }
        for (int i = 0; i < 9; i++) f2 += F[i] * F[i];
        e += h->vol[t] * (0.5 * h->mu * (f2 - 3.0) - h->mu * lj + 0.5 * h->kappa * lj * lj);
    }
    return bad ? INFINITY : e;
}

static double nh_shear_energy(const NH *h, const double *X)
{
    double e = 0;
#pragma omp parallel for reduction(+ : e) schedule(static)
    for (int t = 0; t < h->nt; t++) {
        double F[9]; defgrad(X, h->T + 4 * t, h->gs + 12 * t, F);
        double Bo[9], Q = cauchy_Q(F, Bo);
        e += h->vol[t] * h->mu * Q / det3(F);
    }
    return e;
}

/* volume-weighted rms of the off-diagonal Cauchy stress divided by mu: sqrt(sum vol Q/J / sum vol J) */
static double nh_shear_rms(const NH *h, const double *X)
{
    double num = 0, den = 0;
#pragma omp parallel for reduction(+ : num, den) schedule(static)
    for (int t = 0; t < h->nt; t++) {
        double F[9], Bo[9]; defgrad(X, h->T + 4 * t, h->gs + 12 * t, F);
        double Q = cauchy_Q(F, Bo), J = det3(F);
        num += h->vol[t] * Q / J; den += h->vol[t] * J;
    }
    return sqrt(num / den);
}

static void nh_grad(const NH *h, const double *X, double *out)
{
    memset(out, 0, sizeof(double) * (size_t)h->n);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < h->nt; t++) {
        const int *T4 = h->T + 4 * t; const double *g = h->gs + 12 * t;
        double F[9], Fi[9], P[9];
        defgrad(X, T4, g, F);
        inv3(F, Fi);
        double lj = log(det3(F));
        if (h->mode == 1) {
            double Ft[9], Pc[9];
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Ft[3 * i + j] = Fi[3 * j + i];
            cauchy_P(F, Ft, h->mu, det3(F), Pc);
            for (int i = 0; i < 9; i++) P[i] = Pc[i] + h->kappa * lj * Ft[i];
        } else
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double fit = Fi[3 * j + i];              /* (F^-T)_ij */
                P[3 * i + j] = h->mu * (F[3 * i + j] - fit) + h->kappa * lj * fit;
            }
        for (int a = 0; a < 4; a++)
            for (int i = 0; i < 3; i++) {
                double s = 0;
                for (int j = 0; j < 3; j++) s += P[3 * i + j] * g[3 * a + j];
                s *= h->vol[t];
#pragma omp atomic
                out[3 * T4[a] + i] += s;
            }
    }
}

static void nh_hessian(NH *h, const double *X)
{
    memset(h->va, 0, sizeof(double) * (size_t)h->nnz);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < h->nt; t++) {
        const int *T4 = h->T + 4 * t; const double *g = h->gs + 12 * t;
        double F[9], Fi[9], G[12], gg[16], H4[81];
        defgrad(X, T4, g, F);
        double Jd = inv3(F, Fi);
        if (h->mode == 1) cauchy_H(F, Fi, h->mu, Jd, H4);
        double lj = log(det3(F)), coef = (h->mode == 1 ? 0.0 : h->mu) - h->kappa * lj, vol = h->vol[t];
        for (int a = 0; a < 4; a++)
            for (int i = 0; i < 3; i++) {
                double s = 0;
                for (int j = 0; j < 3; j++) s += Fi[3 * j + i] * g[3 * a + j];
                G[3 * a + i] = s;
            }
        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++)
                gg[4 * a + b] = g[3 * a] * g[3 * b] + g[3 * a + 1] * g[3 * b + 1] + g[3 * a + 2] * g[3 * b + 2];
        for (int a = 0; a < 4; a++) {
            int va_ = T4[a], bpa = h->bp[va_], nbr = h->bp[va_ + 1] - bpa;
            for (int b = 0; b < 4; b++) {
                int kb = h->bpos[16 * t + 4 * a + b] - bpa;
                for (int i = 0; i < 3; i++)
                    for (int k = 0; k < 3; k++) {
                        double v = coef * G[3 * a + k] * G[3 * b + i] + h->kappa * G[3 * a + i] * G[3 * b + k];
                        if (h->mode == 1) {
                            for (int J2 = 0; J2 < 3; J2++)
                                for (int L = 0; L < 3; L++) v += g[3 * a + J2] * g[3 * b + L] * H4[9 * (3 * i + J2) + 3 * k + L];
                        } else if (i == k) v += h->mu * gg[4 * a + b];
                        v *= vol;
                        int pos = 9 * bpa + i * 3 * nbr + 3 * kb + k;
#pragma omp atomic
                        h->va[pos] += v;
                    }
            }
        }
    }
}

static void nh_add_diag(NH *h, const int *b, int nb, const double *k, const double *nrm, double rt)
{
    for (int i = 0; i < nb; i++) {
        int v = b[i], nbr = h->bp[v + 1] - h->bp[v];
        for (int rr = 0; rr < 3; rr++)
            for (int cc = 0; cc < 3; cc++) {
                double add = nrm ? k[i] * ((rt) * (rr == cc ? 1.0 : 0.0) + (1.0 - rt) * nrm[3 * i + rr] * nrm[3 * i + cc]) : (rr == cc ? k[i] : 0.0);
                h->va[9 * h->bp[v] + rr * 3 * nbr + 3 * (h->kd[v] - h->bp[v]) + cc] += add;
            }
    }
}

/* 3x3 block-Jacobi preconditioner; falls back to scalar Jacobi on non-PD blocks */
static void nh_block_prec(const NH *h, double *B)
{
#pragma omp parallel for schedule(static)
    for (int v = 0; v < h->nv; v++) {
        double A[9]; int nbr = h->bp[v + 1] - h->bp[v];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                A[3 * r + c] = h->va[9 * h->bp[v] + r * 3 * nbr + 3 * (h->kd[v] - h->bp[v]) + c];
        int pd = A[0] > 1e-14 * fabs(A[0]) && A[0] > 0;
        double l10 = 0, l20 = 0, l21 = 0, d1 = 0, d2 = 0;
        if (pd) {
            double l00 = sqrt(A[0]);
            l10 = A[3] / l00; d1 = A[4] - l10 * l10;
            if (d1 > 0) {
                l20 = A[6] / l00; l21 = (A[7] - l20 * l10) / sqrt(d1);
                d2 = A[8] - l20 * l20 - l21 * l21;
                if (!(d2 > 0)) pd = 0;
            } else pd = 0;
        }
        double *o = B + 9 * v;
        if (pd) {
            double S[9] = {A[0], 0.5 * (A[1] + A[3]), 0.5 * (A[2] + A[6]),
                           0.5 * (A[1] + A[3]), A[4], 0.5 * (A[5] + A[7]),
                           0.5 * (A[2] + A[6]), 0.5 * (A[5] + A[7]), A[8]};
            inv3(S, o);
        } else {
            for (int i = 0; i < 9; i++) o[i] = 0;
            for (int r = 0; r < 3; r++) { double d = fabs(A[4 * r]); o[4 * r] = 1.0 / (d > 1e-300 ? d : 1.0); }
        }
    }
}

/* conservative alpha<=1 with det(F + alpha dF) > 0 on a sampled grid */
static double nh_max_step(const NH *h, const double *X, const double *d)
{
    const int n = 16; double best = 2.0;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < h->nt; t++) {
        double F[9], dF[9], Fi[9], A[9], A2[9];
        defgrad(X, h->T + 4 * t, h->gs + 12 * t, F);
        defgrad(d, h->T + 4 * t, h->gs + 12 * t, dF);
        inv3(F, Fi);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            double s = 0; for (int k = 0; k < 3; k++) s += Fi[3 * i + k] * dF[3 * k + j];
            A[3 * i + j] = s;
        }
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            double s = 0; for (int k = 0; k < 3; k++) s += A[3 * i + k] * A[3 * k + j];
            A2[3 * i + j] = s;
        }
        double c1 = A[0] + A[4] + A[8], c2 = 0.5 * (c1 * c1 - (A2[0] + A2[4] + A2[8])), c3 = det3(A);
        for (int s = 1; s <= n; s++) {
            double a = (double)s / n;
            if (1.0 + c1 * a + c2 * a * a + c3 * a * a * a <= 0) {
                if (a < best) {
#pragma omp critical
                    { if (a < best) best = a; }
                }
                break;
            }
        }
    }
    return best > 1.5 ? 1.0 : 0.5 * best;
}

static void tet_min_stats(const NH *h, const double *X, const double *V, double *minJ, double *minVol)
{
    double mj = 1e300, mv = 1e300;
    for (int t = 0; t < h->nt; t++) {
        const int *T4 = h->T + 4 * t;
        double F[9]; defgrad(X, T4, h->gs + 12 * t, F);
        double J = det3(F); if (J < mj) mj = J;
        double D[9];
        for (int k = 0; k < 3; k++) for (int i = 0; i < 3; i++) D[3 * i + k] = V[3 * T4[k + 1] + i] - V[3 * T4[0] + i];
        double vv = det3(D) / 6.0; if (vv < mv) mv = vv;
    }
    *minJ = mj; *minVol = mv;
}

/* ------------------------------------------------------------------------ */
/* Newton-CG linear solve: H x = b, truncated at negative curvature         */
/* ------------------------------------------------------------------------ */
static void apply_prec(const double *B, const double *r, double *z, int nv)
{
#pragma omp parallel for schedule(static) if (nv > 4000)
    for (int v = 0; v < nv; v++) {
        const double *b = B + 9 * v, *q = r + 3 * v;
        z[3 * v] = b[0] * q[0] + b[1] * q[1] + b[2] * q[2];
        z[3 * v + 1] = b[3] * q[0] + b[4] * q[1] + b[5] * q[2];
        z[3 * v + 2] = b[6] * q[0] + b[7] * q[1] + b[8] * q[2];
    }
}

static int newton_cg_cpu(const NH *h, const double *B, const double *b, double *x, double tol, int maxit)
{
    int n = h->n, nv = h->nv, k = 0;
    double *w = xmalloc(sizeof(double) * 4 * (size_t)n), *r = w, *z = w + n, *p = z + n, *Ap = p + n;
    memset(x, 0, sizeof(double) * (size_t)n);
    memcpy(r, b, sizeof(double) * (size_t)n);
    apply_prec(B, r, z, nv); memcpy(p, z, sizeof(double) * (size_t)n);
    double rz = dotd(r, z, n), bn = sqrt(dotd(b, b, n));
    if (bn == 0) { free(w); return 0; }
    for (; k < maxit; k++) {
        spmv(n, h->srp, h->sci, h->va, p, Ap);
        double pAp = dotd(p, Ap, n);
        if (!(pAp > 0)) { if (k == 0) memcpy(x, p, sizeof(double) * (size_t)n); break; }
        double al = rz / pAp;
#pragma omp parallel for schedule(static) if (n > 20000)
        for (int i = 0; i < n; i++) { x[i] += al * p[i]; r[i] -= al * Ap[i]; }
        if (sqrt(dotd(r, r, n)) <= tol * bn) { k++; break; }
        apply_prec(B, r, z, nv);
        double rz2 = dotd(r, z, n), be = rz2 / rz; rz = rz2;
#pragma omp parallel for schedule(static) if (n > 20000)
        for (int i = 0; i < n; i++) p[i] = z[i] + be * p[i];
    }
    free(w);
    return k;
}

/* ------------------------------------------------------------------------ */
/* OpenCL backend (float CG; assembly/line search stay in double on host)   */
/* ------------------------------------------------------------------------ */
#ifdef USE_OPENCL
typedef struct {
    int ok, n, nnz, nv, ngroups; size_t local; cl_int err; char name[256];
    cl_context ctx; cl_command_queue q; cl_program prog;
    cl_kernel k_spmv, k_prec, k_dot, k_axpy, k_xpby, k_fill;
    cl_mem rp, ci, va, B, x, r, z, p, Ap, part;
    float *hbuf, *hpart;
} GPU;

static const char *CL_SRC =
"__kernel void spmv(__global const int*rp,__global const int*ci,__global const float*va,\n"
"  __global const float*x,__global float*y,int n){\n"
"  int i=get_global_id(0); if(i>=n) return; float s=0.f;\n"
"  for(int k=rp[i];k<rp[i+1];k++) s+=va[k]*x[ci[k]]; y[i]=s; }\n"
"__kernel void prec(__global const float*B,__global const float*r,__global float*z,int nv){\n"
"  int v=get_global_id(0); if(v>=nv) return; __global const float*b=B+9*v;\n"
"  float r0=r[3*v],r1=r[3*v+1],r2=r[3*v+2];\n"
"  z[3*v]=b[0]*r0+b[1]*r1+b[2]*r2; z[3*v+1]=b[3]*r0+b[4]*r1+b[5]*r2; z[3*v+2]=b[6]*r0+b[7]*r1+b[8]*r2; }\n"
"__kernel void dotk(__global const float*a,__global const float*b,__global float*part,\n"
"  __local float*sh,int n){\n"
"  int lid=get_local_id(0); float s=0.f;\n"
"  for(int i=get_global_id(0);i<n;i+=get_global_size(0)) s+=a[i]*b[i];\n"
"  sh[lid]=s; barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for(int o=get_local_size(0)/2;o>0;o>>=1){ if(lid<o) sh[lid]+=sh[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }\n"
"  if(lid==0) part[get_group_id(0)]=sh[0]; }\n"
"__kernel void axpy(__global float*y,__global const float*x,float a,int n){\n"
"  int i=get_global_id(0); if(i<n) y[i]+=a*x[i]; }\n"
"__kernel void xpby(__global float*y,__global const float*x,float b,int n){\n"
"  int i=get_global_id(0); if(i<n) y[i]=x[i]+b*y[i]; }\n"
"__kernel void fill(__global float*y,float v,int n){\n"
"  int i=get_global_id(0); if(i<n) y[i]=v; }\n";

#define CK(e) do { cl_int _e = (e); if (_e != CL_SUCCESS && g->err == CL_SUCCESS) g->err = _e; } while (0)

static void gpu_list(void)
{
    cl_uint np = 0; clGetPlatformIDs(0, NULL, &np);
    if (!np) { printf("no OpenCL platforms found\n"); return; }
    cl_platform_id *pl = xmalloc(sizeof(cl_platform_id) * np); clGetPlatformIDs(np, pl, NULL);
    for (cl_uint p = 0; p < np; p++) {
        char pn[256] = ""; clGetPlatformInfo(pl[p], CL_PLATFORM_NAME, sizeof pn, pn, NULL);
        cl_uint nd = 0; clGetDeviceIDs(pl[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
        if (!nd) continue;
        cl_device_id *dv = xmalloc(sizeof(cl_device_id) * nd); clGetDeviceIDs(pl[p], CL_DEVICE_TYPE_ALL, nd, dv, NULL);
        for (cl_uint d = 0; d < nd; d++) {
            char dn[256] = ""; cl_device_type ty = 0; cl_uint cu = 0;
            clGetDeviceInfo(dv[d], CL_DEVICE_NAME, sizeof dn, dn, NULL);
            clGetDeviceInfo(dv[d], CL_DEVICE_TYPE, sizeof ty, &ty, NULL);
            clGetDeviceInfo(dv[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
            printf("platform %u (%s)  device %u: %s  [%s, %u CUs]\n", p, pn, d, dn,
                   (ty & CL_DEVICE_TYPE_GPU) ? "GPU" : (ty & CL_DEVICE_TYPE_CPU) ? "CPU" : "other", cu);
        }
        free(dv);
    }
    free(pl);
}

static int gpu_init(GPU *g, int plat, int dev, int force)
{
    memset(g, 0, sizeof *g);
    cl_uint np = 0; clGetPlatformIDs(0, NULL, &np);
    if (!np) return 0;
    cl_platform_id *pl = xmalloc(sizeof(cl_platform_id) * np); clGetPlatformIDs(np, pl, NULL);
    cl_device_id chosen = NULL;
    for (int pass = 0; pass < 2 && !chosen; pass++) {
        if (pass == 1 && !force) break;
        cl_device_type ty = pass == 0 ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL;
        for (cl_uint p = 0; p < np && !chosen; p++) {
            if (plat >= 0 && (int)p != plat) continue;
            cl_uint nd = 0;
            if (clGetDeviceIDs(pl[p], ty, 0, NULL, &nd) != CL_SUCCESS || !nd) continue;
            cl_device_id *dv = xmalloc(sizeof(cl_device_id) * nd); clGetDeviceIDs(pl[p], ty, nd, dv, NULL);
            int di = dev >= 0 ? dev : 0;
            if (di < (int)nd) chosen = dv[di];
            free(dv);
        }
    }
    free(pl);
    if (!chosen) return 0;
    clGetDeviceInfo(chosen, CL_DEVICE_NAME, sizeof g->name, g->name, NULL);
    cl_int e;
    g->ctx = clCreateContext(NULL, 1, &chosen, NULL, NULL, &e); if (e) return 0;
    g->q = clCreateCommandQueue(g->ctx, chosen, 0, &e); if (e) return 0;
    g->prog = clCreateProgramWithSource(g->ctx, 1, &CL_SRC, NULL, &e); if (e) return 0;
    if (clBuildProgram(g->prog, 1, &chosen, "", NULL, NULL) != CL_SUCCESS) {
        char log[4096] = ""; clGetProgramBuildInfo(g->prog, chosen, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "[gpu] kernel build failed:\n%s\n", log); return 0;
    }
    g->k_spmv = clCreateKernel(g->prog, "spmv", &e);  g->k_prec = clCreateKernel(g->prog, "prec", &e);
    g->k_dot = clCreateKernel(g->prog, "dotk", &e);   g->k_axpy = clCreateKernel(g->prog, "axpy", &e);
    g->k_xpby = clCreateKernel(g->prog, "xpby", &e);  g->k_fill = clCreateKernel(g->prog, "fill", &e);
    if (e) return 0;
    size_t maxwg = 128; clGetDeviceInfo(chosen, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof maxwg, &maxwg, NULL);
    g->local = maxwg < 128 ? 64 : 128; g->ngroups = 128;
    g->ok = 1;
    return 1;
}

static int gpu_setup(GPU *g, const NH *h)
{
    cl_int e; g->n = h->n; g->nnz = h->nnz; g->nv = h->nv;
    size_t nb = sizeof(float) * (size_t)g->n;
    g->rp = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ((size_t)g->n + 1), h->srp, &e); if (e) return 0;
    g->ci = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (size_t)g->nnz, h->sci, &e); if (e) return 0;
    g->va = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY, sizeof(float) * (size_t)g->nnz, NULL, &e); if (e) return 0;
    g->B = clCreateBuffer(g->ctx, CL_MEM_READ_ONLY, sizeof(float) * 9 * (size_t)g->nv, NULL, &e); if (e) return 0;
    g->x = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, nb, NULL, &e); if (e) return 0;
    g->r = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, nb, NULL, &e); if (e) return 0;
    g->z = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, nb, NULL, &e); if (e) return 0;
    g->p = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, nb, NULL, &e); if (e) return 0;
    g->Ap = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, nb, NULL, &e); if (e) return 0;
    g->part = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)g->ngroups, NULL, &e); if (e) return 0;
    g->hbuf = xmalloc(sizeof(float) * (size_t)(g->nnz > 9 * g->nv ? g->nnz : 9 * g->nv));
    g->hpart = xmalloc(sizeof(float) * (size_t)g->ngroups);
    return 1;
}

static void g_run(GPU *g, cl_kernel k, size_t n)
{
    CK(clEnqueueNDRangeKernel(g->q, k, 1, NULL, &n, NULL, 0, NULL, NULL));
}
static void g_spmv(GPU *g, cl_mem y, cl_mem x)
{
    cl_int n = g->n;
    CK(clSetKernelArg(g->k_spmv, 0, sizeof(cl_mem), &g->rp)); CK(clSetKernelArg(g->k_spmv, 1, sizeof(cl_mem), &g->ci));
    CK(clSetKernelArg(g->k_spmv, 2, sizeof(cl_mem), &g->va)); CK(clSetKernelArg(g->k_spmv, 3, sizeof(cl_mem), &x));
    CK(clSetKernelArg(g->k_spmv, 4, sizeof(cl_mem), &y));     CK(clSetKernelArg(g->k_spmv, 5, sizeof(cl_int), &n));
    g_run(g, g->k_spmv, (size_t)n);
}
static void g_prec(GPU *g, cl_mem z, cl_mem r)
{
    cl_int nv = g->nv;
    CK(clSetKernelArg(g->k_prec, 0, sizeof(cl_mem), &g->B)); CK(clSetKernelArg(g->k_prec, 1, sizeof(cl_mem), &r));
    CK(clSetKernelArg(g->k_prec, 2, sizeof(cl_mem), &z));    CK(clSetKernelArg(g->k_prec, 3, sizeof(cl_int), &nv));
    g_run(g, g->k_prec, (size_t)nv);
}
static double g_dot(GPU *g, cl_mem a, cl_mem b)
{
    cl_int n = g->n; size_t gs = (size_t)g->ngroups * g->local, ls = g->local;
    CK(clSetKernelArg(g->k_dot, 0, sizeof(cl_mem), &a)); CK(clSetKernelArg(g->k_dot, 1, sizeof(cl_mem), &b));
    CK(clSetKernelArg(g->k_dot, 2, sizeof(cl_mem), &g->part));
    CK(clSetKernelArg(g->k_dot, 3, sizeof(float) * ls, NULL)); CK(clSetKernelArg(g->k_dot, 4, sizeof(cl_int), &n));
    CK(clEnqueueNDRangeKernel(g->q, g->k_dot, 1, NULL, &gs, &ls, 0, NULL, NULL));
    CK(clEnqueueReadBuffer(g->q, g->part, CL_TRUE, 0, sizeof(float) * (size_t)g->ngroups, g->hpart, 0, NULL, NULL));
    double s = 0; for (int i = 0; i < g->ngroups; i++) s += g->hpart[i];
    return s;
}
static void g_axpy(GPU *g, cl_mem y, cl_mem x, float a)
{
    cl_int n = g->n;
    CK(clSetKernelArg(g->k_axpy, 0, sizeof(cl_mem), &y)); CK(clSetKernelArg(g->k_axpy, 1, sizeof(cl_mem), &x));
    CK(clSetKernelArg(g->k_axpy, 2, sizeof(float), &a));  CK(clSetKernelArg(g->k_axpy, 3, sizeof(cl_int), &n));
    g_run(g, g->k_axpy, (size_t)n);
}
static void g_xpby(GPU *g, cl_mem y, cl_mem x, float b)
{
    cl_int n = g->n;
    CK(clSetKernelArg(g->k_xpby, 0, sizeof(cl_mem), &y)); CK(clSetKernelArg(g->k_xpby, 1, sizeof(cl_mem), &x));
    CK(clSetKernelArg(g->k_xpby, 2, sizeof(float), &b));  CK(clSetKernelArg(g->k_xpby, 3, sizeof(cl_int), &n));
    g_run(g, g->k_xpby, (size_t)n);
}

/* returns iterations, or -1 on any failure (caller falls back to CPU) */
static int newton_cg_gpu(GPU *g, const NH *h, const double *B, const double *b, double *x,
                         double tol, int maxit, Timers *tm)
{
    double t0 = now(); g->err = CL_SUCCESS;
    int n = g->n;
    for (int i = 0; i < g->nnz; i++) g->hbuf[i] = (float)h->va[i];
    CK(clEnqueueWriteBuffer(g->q, g->va, CL_FALSE, 0, sizeof(float) * (size_t)g->nnz, g->hbuf, 0, NULL, NULL));
    float *tmp = xmalloc(sizeof(float) * 9 * (size_t)g->nv);
    for (int i = 0; i < 9 * g->nv; i++) tmp[i] = (float)B[i];
    CK(clEnqueueWriteBuffer(g->q, g->B, CL_TRUE, 0, sizeof(float) * 9 * (size_t)g->nv, tmp, 0, NULL, NULL));
    for (int i = 0; i < n; i++) g->hbuf[i] = (float)b[i];
    CK(clEnqueueWriteBuffer(g->q, g->r, CL_TRUE, 0, sizeof(float) * (size_t)n, g->hbuf, 0, NULL, NULL));
    free(tmp);
    tm->gpu_up += now() - t0;

    cl_int ni = n; float zero = 0.f;
    CK(clSetKernelArg(g->k_fill, 0, sizeof(cl_mem), &g->x)); CK(clSetKernelArg(g->k_fill, 1, sizeof(float), &zero));
    CK(clSetKernelArg(g->k_fill, 2, sizeof(cl_int), &ni)); g_run(g, g->k_fill, (size_t)n);
    g_prec(g, g->z, g->r);
    CK(clEnqueueCopyBuffer(g->q, g->z, g->p, 0, 0, sizeof(float) * (size_t)n, 0, NULL, NULL));
    double rz = g_dot(g, g->r, g->z), bn = sqrt(g_dot(g, g->r, g->r));
    int k = 0;
    if (!(bn > 0)) { memset(x, 0, sizeof(double) * (size_t)n); return g->err ? -1 : 0; }
    for (; k < maxit && !g->err; k++) {
        g_spmv(g, g->Ap, g->p);
        double pAp = g_dot(g, g->p, g->Ap);
        if (!(pAp > 0)) { if (k == 0) CK(clEnqueueCopyBuffer(g->q, g->p, g->x, 0, 0, sizeof(float) * (size_t)n, 0, NULL, NULL)); break; }
        float al = (float)(rz / pAp);
        g_axpy(g, g->x, g->p, al); g_axpy(g, g->r, g->Ap, -al);
        double rn = sqrt(g_dot(g, g->r, g->r));
        if (rn <= tol * bn) { k++; break; }
        g_prec(g, g->z, g->r);
        double rz2 = g_dot(g, g->r, g->z);
        g_xpby(g, g->p, g->z, (float)(rz2 / rz)); rz = rz2;
    }
    CK(clEnqueueReadBuffer(g->q, g->x, CL_TRUE, 0, sizeof(float) * (size_t)n, g->hbuf, 0, NULL, NULL));
    if (g->err) return -1;
    for (int i = 0; i < n; i++) { x[i] = g->hbuf[i]; if (!isfinite(x[i])) return -1; }
    return k;
}
#else
typedef struct { int ok; } GPU;
#endif

/* ------------------------------------------------------------------------ */
/* Newton relaxation                                                         */
/* ------------------------------------------------------------------------ */
static double total_energy(const NH *nh, const double *X, const int *b, int nb,
                           const double *target, const double *kfit, const double *nrm, double rt)
{
    double e = nh_energy(nh, X);
    if (!isfinite(e)) return INFINITY;
    double p = 0;
    for (int i = 0; i < nb; i++) {
        const double *x = X + 3 * b[i], *t = target + 3 * i;
        if (nrm) {
            double d = nrm[3 * i] * (x[0] - t[0]) + nrm[3 * i + 1] * (x[1] - t[1]) + nrm[3 * i + 2] * (x[2] - t[2]);
            p += kfit[i] * (rt * ((x[0] - t[0]) * (x[0] - t[0]) + (x[1] - t[1]) * (x[1] - t[1]) + (x[2] - t[2]) * (x[2] - t[2])) + (1.0 - rt) * d * d);
        } else
            p += kfit[i] * ((x[0] - t[0]) * (x[0] - t[0]) + (x[1] - t[1]) * (x[1] - t[1]) + (x[2] - t[2]) * (x[2] - t[2]));
    }
    return e + 0.5 * p;
}

static int relax(double *V, const double *target, const int *b, int nb, const double *kfit, NH *nh,
                 const double *nrm, double rt, const double *rc, double rR, const Mesh *mesh, const Args *A, double h, GPU *gp, Timers *tm)
{
    (void)mesh;
    int n = nh->n, moved = 0;   /* rc != NULL: retract boundary onto sphere (rc, rR) in every trial; penalty not in the energy */
    double *w = xmalloc(sizeof(double) * 4 * (size_t)n), *g = w, *rhs = g + n, *d = rhs + n, *Vt = d + n;
    double *B = xmalloc(sizeof(double) * 9 * (size_t)nh->nv);
    int use_gpu = 0; (void)use_gpu;
#ifdef USE_OPENCL
    use_gpu = gp && gp->ok && A->gpu && n >= A->gpu_min_rows;
#else
    (void)gp;
#endif
    for (int it = 0; it < A->newton_iters; it++) {
        nh_grad(nh, V, g);
        for (int i = 0; i < nb; i++) {
            const double *x = V + 3 * b[i], *t = target + 3 * i;
            if (nrm) {
                double d = nrm[3 * i] * (x[0] - t[0]) + nrm[3 * i + 1] * (x[1] - t[1]) + nrm[3 * i + 2] * (x[2] - t[2]);
                for (int c = 0; c < 3; c++) g[3 * b[i] + c] += kfit[i] * (rt * (x[c] - t[c]) + (1.0 - rt) * nrm[3 * i + c] * d);
            } else
                for (int c = 0; c < 3; c++) g[3 * b[i] + c] += kfit[i] * (x[c] - t[c]);
        }
        double t0 = now();
        nh_hessian(nh, V); nh_add_diag(nh, b, nb, kfit, nrm, rt);
        double t1 = now(); nh_block_prec(nh, B);
        double t2 = now(); tm->asmb += t1 - t0; tm->prec += t2 - t1;
        for (int i = 0; i < n; i++) rhs[i] = -g[i];

        int its = -1;
#ifdef USE_OPENCL
        if (use_gpu) {
            its = newton_cg_gpu(gp, nh, B, rhs, d, A->cg_tol, A->cg_iters, tm);
            if (its >= 0) tm->gpu_solves++;
        }
#endif
        if (its < 0) { its = newton_cg_cpu(nh, B, rhs, d, A->cg_tol, A->cg_iters); tm->cpu_solves++; }
        tm->cg += now() - t2; tm->cg_its += its;

        double t3 = now();
        double e0 = rc ? nh_energy(nh, V) : total_energy(nh, V, b, nb, target, kfit, nrm, rt);
        double slope = dotd(g, d, n);
        if (!(slope < 0)) break;
        double alpha = nh_max_step(nh, V, d), dmax = 0;
        for (int i = 0; i < n; i++) if (fabs(d[i]) > dmax) dmax = fabs(d[i]);
        int ok = 0;
        for (int ls = 0; ls < A->max_ls; ls++) {
            for (int i = 0; i < n; i++) Vt[i] = V[i] + alpha * d[i];
            if (rc)
                for (int i = 0; i < nb; i++) {
                    double *x = Vt + 3 * b[i], dd[3] = {x[0] - rc[0], x[1] - rc[1], x[2] - rc[2]};
                    double r = sqrt(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]); if (r < 1e-300) r = 1;
                    for (int k = 0; k < 3; k++) x[k] = rc[k] + rR * dd[k] / r;
                }
            double et = rc ? nh_energy(nh, Vt) : total_energy(nh, Vt, b, nb, target, kfit, nrm, rt);
            if (et <= e0 + 1e-4 * alpha * slope) { ok = 1; break; }
            alpha *= 0.5;
        }
        tm->ls += now() - t3;
        if (!ok) break;
        memcpy(V, Vt, sizeof(double) * (size_t)n); moved++;
        if (alpha * dmax < A->newton_tol * h) break;
    }
    free(w); free(B);
    return moved;
}

/* ------------------------------------------------------------------------ */
/* video: frames are streamed to render_video.py through a pipe (no files)  */
/* ------------------------------------------------------------------------ */
static char g_exe_dir[1024] = ".";
#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#define POPEN_MODE "wb"
#else
#define POPEN popen
#define PCLOSE pclose
#define POPEN_MODE "w"
#endif
static FILE *video_open(const Args *A, const Mesh *m)
{
    char script[1200], cmd[4096];
    if (A->script) snprintf(script, sizeof script, "%s", A->script);
    else snprintf(script, sizeof script, "%s/render_video.py", g_exe_dir);
    snprintf(cmd, sizeof cmd, "%s \"%s\" \"%s\" %d %s %d %d", A->python, script, A->video, A->fps, A->vsize,
             A->xvfb, A->steps);
    FILE *f = POPEN(cmd, POPEN_MODE);
    if (!f) { p_warn("cannot start video renderer: %s", cmd); return NULL; }
    int32_t hdr[2] = {m->nb, m->nf};
    fwrite(hdr, 4, 2, f);
    fwrite(m->Fl, 4, 3 * (size_t)m->nf, f);
    p_info("video \u2192 %s", A->video);
    return f;
}
static void video_frame(FILE **f, int step, double dev, const double *S, int nb)
{
    if (!*f) return;
    float *buf = xmalloc(sizeof(float) * 3 * (size_t)nb);
    for (int i = 0; i < 3 * nb; i++) buf[i] = (float)S[i];
    int32_t st = step;
    int ok = fwrite(&st, 4, 1, *f) == 1 && fwrite(&dev, 8, 1, *f) == 1 &&
             fwrite(buf, 4, 3 * (size_t)nb, *f) == 3 * (size_t)nb && fflush(*f) == 0;
    free(buf);
    if (!ok) { live_erase(); p_warn("video renderer stopped; continuing without video"); live_draw(); PCLOSE(*f); *f = NULL; }
}

/* Move boundary vertices toward the sphere (centre c, radius R): largest fraction
 * lambda in {1, 1/2, 1/4, ...} that keeps every J > 0.  Returns lambda (0 = none). */
static double snap_sphere(const Mesh *m, const NH *nh, const double *V, double *Vout, const double *c, double R)
{
    double lam = 1.0;
    for (int tr = 0; tr < 8; tr++, lam *= 0.5) {
        memcpy(Vout, V, sizeof(double) * 3 * (size_t)m->nv);
        for (int i = 0; i < m->nb; i++) {
            double *x = Vout + 3 * m->b[i], d[3] = {x[0] - c[0], x[1] - c[1], x[2] - c[2]};
            double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]); if (r < 1e-300) r = 1;
            for (int k = 0; k < 3; k++) x[k] += lam * (c[k] + R * d[k] / r - x[k]);
        }
        if (isfinite(nh_energy(nh, Vout))) return lam;
    }
    memcpy(Vout, V, sizeof(double) * 3 * (size_t)m->nv);
    return 0.0;
}
static double sphere_dev(const Mesh *m, const double *V, const double *c, double R)
{
    double mx = 0;
    for (int i = 0; i < m->nb; i++) {
        const double *x = V + 3 * m->b[i];
        double r = sqrt((x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1]) + (x[2] - c[2]) * (x[2] - c[2]));
        if (fabs(r - R) / R > mx) mx = fabs(r - R) / R;
    }
    return mx;
}

/* ------------------------------------------------------------------------ */
/* flow                                                                      */
/* ------------------------------------------------------------------------ */
/* live two-row display for phase 2 (bar + self-replacing stats) */
static void shear_bar(const NH *nh, const double *V, const Timers *tm, int it2, int total,
                      double E, const char *extra, double elapsed)
{
    if (!g_live || !total) return;
    double mj = 1e300, mv = 1e300;
    tet_min_stats(nh, V, V, &mj, &mv);
    char bar[1000], r1[1600], r2[900], eta[24];
    bar_render(bar, sizeof bar, (double)(it2 + 1) / total, 32);
    fmt_t(it2 > 0 ? elapsed * (total - it2 - 1) / (it2 + 1) : 0, eta, sizeof eta);
    snprintf(r1, sizeof r1, "%s%s %s phase 2%s  %s %d/%d  E=%.4e  %s  %02d:%02d / ETA %s",
             g_color ? C_PINK : "", SPIN[it2 % 10], g_color ? ESC_R : "", "",
             bar, it2 + 1, total, E, extra, (int)elapsed / 60, (int)elapsed % 60, eta);
    snprintf(r2, sizeof r2, "%s\u23f1%s assemble %.2f \u00b7 CG %.2f (%ld its, %d solves) \u00b7 line-search %.2f \u00b7 minJ %.2e",
             g_color ? C_SOFT : "", g_color ? ESC_R : "",
             tm->asmb, tm->cg, tm->cg_its, tm->gpu_solves + tm->cpu_solves, tm->ls, mj);
    live_set(r1, r2);
}

static double radius_dev(const double *S, int nb)
{
    double mean[3] = {0, 0, 0}, rm = 0, r2 = 0;
    for (int i = 0; i < nb; i++) for (int k = 0; k < 3; k++) mean[k] += S[3 * i + k] / nb;
    for (int i = 0; i < nb; i++) {
        double d0 = S[3 * i] - mean[0], d1 = S[3 * i + 1] - mean[1], d2 = S[3 * i + 2] - mean[2];
        double r = sqrt(d0 * d0 + d1 * d1 + d2 * d2); rm += r; r2 += r * r;
    }
    rm /= nb; r2 = r2 / nb - rm * rm;
    return sqrt(r2 > 0 ? r2 : 0) / (rm + 1e-30);
}

/* ------------------------------------------------------------------------ */
/* stress fields on the sphere                                               */
/* ------------------------------------------------------------------------ */
/* Full Neo-Hookean Cauchy stress of the final map (reference = initial shape):
 *   sigma = (mu (b - I) + kappa ln J I)/J,   b = F F^T
 * volume-weighted over the tets adjacent to each boundary vertex:
 *   p        = tr(sigma)/3                                 (scalar pressure)
 *   deviator = (sigma_xx - p, sigma_yy - p, sigma_zz - p)  (on-diagonal traceless)
 * Written as legacy VTK POLYDATA on the flowed boundary (the sphere). */
static void write_stress_fields(const NH *nh, const Mesh *m, const double *V,
                                double mu, double kappa, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    double *sp = xcalloc((size_t)m->nv, sizeof(double));   /* sum w*p        */
    double *sd = xcalloc(3 * (size_t)m->nv, sizeof(double)); /* sum w*dev     */
    double *sw = xcalloc((size_t)m->nv, sizeof(double));   /* sum w          */
    int *loc = xmalloc(sizeof(int) * (size_t)m->nv);       /* global -> sphere-local */
    for (int i = 0; i < m->nv; i++) loc[i] = -1;
    for (int i = 0; i < m->nb; i++) loc[m->b[i]] = i;
    for (int t = 0; t < nh->nt; t++) {
        const int *T4 = nh->T + 4 * t;
        double F[9]; defgrad(V, T4, nh->gs + 12 * t, F);
        double J = det3(F);
        if (!(J > 0)) continue;
        double lj = log(J), b[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double s = 0;
                for (int k = 0; k < 3; k++) s += F[3 * i + k] * F[3 * j + k];
                b[3 * i + j] = s;
            }
        double tr = b[0] + b[4] + b[8];
        double p = (mu * (tr - 3.0) + 3.0 * kappa * lj) / (3.0 * J);
        double d[3];
        for (int k = 0; k < 3; k++) d[k] = (mu * (b[3 * k + k] - 1.0) + kappa * lj) / J - p;
        double w = nh->vol[t] * J;                        /* current tet volume */
        for (int a = 0; a < 4; a++) {
            int v = T4[a];
            if (loc[v] < 0) continue;
            sp[v] += w * p;
            for (int k = 0; k < 3; k++) sd[3 * v + k] += w * d[k];
            sw[v] += w;
        }
    }
    fprintf(f, "# vtk DataFile Version 3.0\nmiso stress on the sphere\nASCII\n");
    fprintf(f, "DATASET POLYDATA\nPOINTS %d double\n", m->nb);
    for (int i = 0; i < m->nb; i++) {
        const double *x = V + 3 * m->b[i];
        fprintf(f, "%.10g %.10g %.10g\n", x[0], x[1], x[2]);
    }
    fprintf(f, "POLYGONS %d %d\n", m->nf, 4 * m->nf);
    for (int i = 0; i < m->nf; i++)
        fprintf(f, "3 %d %d %d\n", loc[m->F[3 * i]], loc[m->F[3 * i + 1]], loc[m->F[3 * i + 2]]);
    fprintf(f, "POINT_DATA %d\nSCALARS pressure double 1\nLOOKUP_TABLE default\n", m->nb);
    for (int i = 0; i < m->nb; i++) {
        int v = m->b[i];
        fprintf(f, "%.10g\n", sw[v] > 0 ? sp[v] / sw[v] : 0.0);
    }
    fprintf(f, "VECTORS deviatoric double\n");
    for (int i = 0; i < m->nb; i++) {
        int v = m->b[i];
        double w = sw[v] > 0 ? 1.0 / sw[v] : 0.0;
        fprintf(f, "%.10g %.10g %.10g\n", sd[3 * v] * w, sd[3 * v + 1] * w, sd[3 * v + 2] * w);
    }
    fclose(f);
    free(sp); free(sd); free(sw); free(loc);
}

static void flow(Mesh *m, double *V, const Args *A, GPU *gp)
{
    int nv = m->nv, nb = m->nb, nf = m->nf; const int has_tet = m->T != NULL;
    double *S = xmalloc(sizeof(double) * 3 * (size_t)nb), *U = xmalloc(sizeof(double) * 3 * (size_t)nb);
    double *mass = xmalloc(sizeof(double) * (size_t)nb), *kfit = xmalloc(sizeof(double) * (size_t)nb);
    double *nrm = xmalloc(sizeof(double) * 3 * (size_t)nb);
    double *Vold = xmalloc(sizeof(double) * 3 * (size_t)nv), *V0 = xmalloc(sizeof(double) * 3 * (size_t)nv);
    double *Vref0 = xmalloc(sizeof(double) * 3 * (size_t)nv);   /* initial shape, tracked through rescaling */
    Timers tm; memset(&tm, 0, sizeof tm);

    memcpy(V0, V, sizeof(double) * 3 * (size_t)nv); memcpy(Vref0, V, sizeof(double) * 3 * (size_t)nv);
    gather(V, m->b, nb, S);
    double h = mean_edge(S, m->Fl, nf), delta = A->delta_scale * h * h, area0 = surf_area(S, m->Fl, nf);
    double target_area = area0;
    if (A->target_vol) {
        double vol0 = signed_volume(V, m->F, nf);
        target_area = 4.0 * M_PI * pow(3.0 * vol0 / (4.0 * M_PI), 2.0 / 3.0);
    }
    double k_scale = A->fit * (A->mu + A->kappa) / h;
    p_info("mean_edge=%.4e  delta=%.4e  area=%.5f  target=%.5f", h, delta, area0, target_area);

    SurfOp so; surf_init(&so, m->Fl, nf, nb);
    NH nh; double init_min_vol = 1;
    if (has_tet) {
        nh_init(&nh, m->T, m->nt, nv, A->mu, A->kappa); nh_set_ref(&nh, V0);
        double mj, mv; tet_min_stats(&nh, V, V, &mj, &mv); init_min_vol = mv;
        p_info("NH: mu=%g kappa=%g fit=%g ref=%s  tets=%d  system rows=%d nnz=%d", A->mu, A->kappa, A->fit,
               A->ref_updated ? "updated" : "initial", m->nt, nh.n, nh.nnz);
        p_info("init min_vol=%.3e", mv);
#ifdef USE_OPENCL
        if (gp && gp->ok) {
            if (gpu_setup(gp, &nh)) {
                p_info("OpenCL device: %s  (used when rows >= %d)", gp->name, A->gpu_min_rows);
            } else { gp->ok = 0; p_warn("OpenCL buffer setup failed, CPU only"); }
        }
#endif
    }

    double c[3] = {0, 0, 0}; for (int i = 0; i < nb; i++) for (int k = 0; k < 3; k++) c[k] += S[3 * i + k] / nb;
    int radial = 0, cooldown_until = 0, converged = 0;
    double *devh = xcalloc((size_t)A->steps + 1, sizeof(double));
    FILE *vid = A->no_video ? NULL : video_open(A, m);
    video_frame(&vid, 0, 0.0, S, nb);
    double tflow = now();

    /* ======================= phase 1: boundary to a sphere ======================= */
    for (int it = 0; it < A->steps; it++) {
        memcpy(Vold, V, sizeof(double) * 3 * (size_t)nv);
        gather(Vold, m->b, nb, S);

        double t0 = now();
        if (A->radial) {
            /* enter with a safety margin (1% of an edge), stay with a loose one */
            double thr = radial ? 1e-6 : 1e-2;
            int star = star_margin(S, m->Fl, nf, c) > thr * h;
            if (!radial && it < cooldown_until) thr = 1e30;   /* just lost it: wait */
            if (!star && thr < 1e29 && (radial || it % A->star_every == 0))
                star = find_kernel_point(S, m->Fl, nf, nb, h, c, radial ? 30 : 60, thr);
            if (star && !radial) {
                live_erase();
                p_star("STAR DOMAIN \u2728 kernel (%.3g, %.3g, %.3g) \u2014 radial projection flow", c[0], c[1], c[2]);
                live_draw();
            }
            if (!star && radial) {
                live_erase();
                p_warn("lost star-shapedness at step %d: back to cMCF (retry in 10 steps)", it + 1);
                live_draw();
                cooldown_until = it + 10;
            }
            radial = star;
        }
        tm.star += now() - t0;

        t0 = now();
        if (radial) {
            radial_target(S, nb, c, sqrt(target_area / (4.0 * M_PI)), A->radial_rate, U);
        } else {
            lumped_mass(S, m->Fl, nf, nb, mass);
            if (!cmcf_step(&so, S, m->Fl, delta, A->min_cot, mass, U)) { live_end(); p_err("cMCF failed at step %d", it + 1); break; }
        }
        tm.cmcf += now() - t0;

        int moved = 0;
        if (has_tet) {
            lumped_mass(S, m->Fl, nf, nb, mass);
            for (int i = 0; i < nb; i++) kfit[i] = k_scale * mass[i];
            moved = relax(V, U, m->b, nb, kfit, &nh, NULL, 1.0, NULL, 0.0, m, A, h, gp, &tm);
            if (moved == 0) { live_end(); p_err("Newton made no progress at step %d", it + 1); break; }
        } else {
            for (int i = 0; i < nb; i++) memcpy(V + 3 * m->b[i], U + 3 * i, 3 * sizeof(double));
        }

        gather(V, m->b, nb, S);
        double mean[3] = {0, 0, 0};
        for (int i = 0; i < nb; i++) for (int k = 0; k < 3; k++) mean[k] += S[3 * i + k] / nb;
        for (int i = 0; i < nv; i++) for (int k = 0; k < 3; k++) V[3 * i + k] -= mean[k];
        for (int k = 0; k < 3; k++) c[k] -= mean[k];
        gather(V, m->b, nb, S);
        double s = sqrt(target_area / surf_area(S, m->Fl, nf));
        for (int i = 0; i < 3 * nv; i++) V[i] *= s;
        for (int k = 0; k < 3; k++) c[k] *= s;
        gather(V, m->b, nb, S);
        if (has_tet) {
            for (int i = 0; i < 3 * nv; i++) Vref0[i] *= s;
            if (A->ref_updated) memcpy(V0, V, sizeof(double) * 3 * (size_t)nv);
            else memcpy(V0, Vref0, sizeof(double) * 3 * (size_t)nv);
            nh_set_ref(&nh, V0);
        }

        double dev = radius_dev(S, nb);
        double elapsed = now() - tflow;
        if (g_live) {
            double mj = 0, mv = 0;
            if (has_tet) tet_min_stats(&nh, V, V, &mj, &mv);
            char bar[1000], r1[1600], r2[900], eta[24], minjs[48] = "";
            bar_render(bar, sizeof bar, (double)(it + 1) / A->steps, 32);
            fmt_t(it > 0 ? elapsed * (A->steps - it - 1) / (it + 1) : 0, eta, sizeof eta);
            if (has_tet) snprintf(minjs, sizeof minjs, "  \u00b7  minJ %.2e", mj);
            snprintf(r1, sizeof r1, "%s%s %s phase 1%s  %s %d/%d  dev=%.3e  %02d:%02d / ETA %s",
                     g_color ? C_PINK : "", SPIN[it % 10], g_color ? ESC_R : "", "",
                     bar, it + 1, A->steps, dev, (int)elapsed / 60, (int)elapsed % 60, eta);
            if (has_tet)
                snprintf(r2, sizeof r2,
                         "%s\u23f1%s star %.2f \u00b7 target %.2f \u00b7 assemble %.2f \u00b7 CG %.2f (%ld its) \u00b7 line-search %.2f%s",
                         g_color ? C_SOFT : "", g_color ? ESC_R : "",
                         tm.star, tm.cmcf, tm.asmb, tm.cg, tm.cg_its, tm.ls, minjs);
            else
                snprintf(r2, sizeof r2, "%s\u23f1%s star %.2f \u00b7 target %.2f  (surface flow)",
                         g_color ? C_SOFT : "", g_color ? ESC_R : "", tm.star, tm.cmcf);
            live_set(r1, r2);
        } else if (it % 5 == 0 || it == A->steps - 1 || dev < A->tol) {
            char extra[160] = "";
            if (has_tet) {
                double mj, mv; tet_min_stats(&nh, V, V, &mj, &mv);
                snprintf(extra, sizeof extra, "  newton=%2d  minJ=%.3e  minV=%.2e (%.0f%%)", moved, mj, mv, 100.0 * mv / init_min_vol);
            }
            p_step("step %d/%d  [%s]  dev=%.3e%s", it + 1, A->steps, radial ? "radial" : "cMCF", dev, extra);
        }
        if (((it + 1) % A->video_every == 0 || dev < A->tol)) video_frame(&vid, it + 1, dev, S, nb);
        devh[it] = dev;
        if (dev < A->tol) { live_end(); p_ok("boundary is a sphere at step %d, dev=%.3e", it + 1, dev); converged = 1; break; }
        if (has_tet && A->shear && radial && it >= 10 && dev > 0.98 * devh[it - 10] && dev < 0.1) {
            live_end();
            p_ok("radial flow stalled at step %d (dev=%.3e, elastic pull-back) \u2014 phase 2 pins the boundary to the sphere", it + 1, dev);
            converged = 1; break;
        }
    }
    live_end();
    p_ok("phase 1: %.2fs | star %.2f  target %.2f  assemble %.2f  precond %.2f  CG %.2f (%ld its, %d GPU / %d CPU solves)  line-search %.2f",
         now() - tflow, tm.star, tm.cmcf, tm.asmb, tm.prec, tm.cg, tm.cg_its, tm.gpu_solves, tm.cpu_solves, tm.ls);

    /* ============ phase 2: relax tets with off-diagonal Cauchy stress only ============ */
    if (has_tet && A->shear) {
        if (!converged) {
            gather(V, m->b, nb, S);
            p_warn("boundary did not reach a sphere within %d steps (dev=%.3e): phase 2 approaches it directly",
                   A->steps, radius_dev(S, nb));
        }
        {
            double t2 = now();
            nh.mode = 1; nh.mu = A->mu; nh.kappa = A->kappa_shear;
            nh_set_ref(&nh, Vref0);               /* absolute reference: the initial shape */
            double mj, mv; tet_min_stats(&nh, V, V, &mj, &mv);
            double R = sqrt(target_area / (4.0 * M_PI)), k2 = A->fit_shear * (A->mu + A->kappa) / h;
            gather(V, m->b, nb, S);
            for (int k = 0; k < 3; k++) c[k] = 0;
            for (int i = 0; i < nb; i++) for (int k = 0; k < 3; k++) c[k] += S[3 * i + k] / nb;
            double E0 = nh_shear_energy(&nh, V), rms0 = nh_shear_rms(&nh, V);
            double *Vs = xmalloc(sizeof(double) * 3 * (size_t)nv), *Vsave = xmalloc(sizeof(double) * 3 * (size_t)nv);
            p_hdr("phase 2 \u2014 off-diagonal Cauchy stress only: shear energy %.6e  minJ=%.3e", E0, mj);
            double Etot = nh_energy(&nh, V), Es0 = E0, devPrev = sphere_dev(m, V, c, R);
            int onsph = devPrev < 1e-8;
            p_info("boundary radial deviation %.2e: %s", devPrev, onsph ? "on the sphere" : "approaching the sphere in J>0-safe steps");
            double rt = A->shear_tang; int it2 = 0, minimum = 0, rejected = 0;
            for (; it2 < A->shear_steps; it2++) {
                memcpy(Vsave, V, sizeof(double) * 3 * (size_t)nv);
                gather(V, m->b, nb, S);
                for (int i = 0; i < nb; i++) {
                    double d[3] = {S[3 * i] - c[0], S[3 * i + 1] - c[1], S[3 * i + 2] - c[2]};
                    double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]); if (r < 1e-300) r = 1;
                    for (int k = 0; k < 3; k++) { nrm[3 * i + k] = d[k] / r; U[3 * i + k] = c[k] + R * d[k] / r; }
                }
                lumped_mass(S, m->Fl, nf, nb, mass);
                if (onsph) {
                    /* constrained step: sphere-normal motion blocked stiffly, tangential motion free,
                     * every line-search trial retracted exactly onto the sphere; energy decreases monotonically */
                    for (int i = 0; i < nb; i++) kfit[i] = 100.0 * k2 * mass[i];
                    int moved = relax(V, U, m->b, nb, kfit, &nh, nrm, 0.0, c, R, m, A, h, gp, &tm);
                    double En = nh_energy(&nh, V);
                    if (moved == 0 || !(En <= Etot)) { memcpy(V, Vsave, sizeof(double) * 3 * (size_t)nv); minimum = 1; break; }
                    double rel = (Etot - En) / (Etot > 1e-300 ? Etot : 1e-300); Etot = En;
                    double E = nh_shear_energy(&nh, V);
                    gather(V, m->b, nb, S);
                    if (g_live) {
                        char ex[96];
                        snprintf(ex, sizeof ex, "\u0394E=%.2e  dev=%.1e", rel, radius_dev(S, nb));
                        shear_bar(&nh, V, &tm, it2, A->shear_steps, E, ex, now() - t2);
                    } else if (it2 % 10 == 0 || rel < A->shear_tol) {
                        tet_min_stats(&nh, V, V, &mj, &mv);
                        p_step("shear %d/%d  E_shear=%.6e  rel.change=%.2e  newton=%d  minJ=%.3e",
                               it2 + 1, A->shear_steps, E, rel, moved, mj);
                    }
                    if ((it2 + 1) % A->video_every == 0) video_frame(&vid, -(it2 + 1), E / (E0 > 0 ? E0 : 1), S, nb);
                    if (rel < A->shear_tol) { minimum = 1; it2++; break; }
                } else {
                    /* approach: soft penalty + largest J>0-safe partial snap toward the sphere */
                    for (int i = 0; i < nb; i++) kfit[i] = k2 * mass[i];
                    int moved = relax(V, U, m->b, nb, kfit, &nh, nrm, rt, NULL, 0.0, m, A, h, gp, &tm);
                    if (moved == 0) { memcpy(V, Vsave, sizeof(double) * 3 * (size_t)nv); live_end();
                        p_err("Newton made no progress in phase 2 at iteration %d", it2 + 1); break; }
                    double lam = snap_sphere(m, &nh, V, Vs, c, R), dS = lam > 0 ? sphere_dev(m, Vs, c, R) : 1e30;
                    if (lam > 0 && dS < 0.999 * devPrev) {
                        memcpy(V, Vs, sizeof(double) * 3 * (size_t)nv); devPrev = dS; Etot = nh_energy(&nh, V);
                        if (dS < 1e-8) {
                            onsph = 1; Es0 = nh_shear_energy(&nh, V); rms0 = nh_shear_rms(&nh, V);
                            live_erase();
                            p_ok("boundary is on the sphere at shear iteration %d \u2014 shear energy %.6e, now driving \u0394E \u2192 0", it2 + 1, Es0);
                            live_draw();
                        }
                    } else { memcpy(V, Vsave, sizeof(double) * 3 * (size_t)nv); rt *= 4; rejected++; if (rt > 2.0) rt = 2.0; }
                    gather(V, m->b, nb, S);
                    if (g_live) {
                        char ex[96];
                        snprintf(ex, sizeof ex, "dev=%.1e  (approach rt=%.2f)", devPrev, rt);
                        shear_bar(&nh, V, &tm, it2, A->shear_steps, nh_shear_energy(&nh, V), ex, now() - t2);
                    } else if (it2 % 10 == 0) {
                        p_step("shear %d/%d  (approach)  dev=%.3e  rt=%.2f  rejected=%d",
                               it2 + 1, A->shear_steps, devPrev, rt, rejected);
                    }
                    if ((it2 + 1) % A->video_every == 0) video_frame(&vid, -(it2 + 1), nh_shear_energy(&nh, V) / (E0 > 0 ? E0 : 1), S, nb);
                }
            }
            live_end();
            p_info("rms off-diagonal Cauchy stress / mu: %.4e (on the sphere) -> %.4e (final)", rms0, nh_shear_rms(&nh, V));
            if (!onsph) p_warn("the boundary could not be brought exactly onto the sphere without inverting a tet (dev %.2e)", devPrev);
            free(Vs); free(Vsave);
            gather(V, m->b, nb, S);
            double Efin = nh_shear_energy(&nh, V); tet_min_stats(&nh, V, V, &mj, &mv);
            video_frame(&vid, -(it2 + 1), Efin / (E0 > 0 ? E0 : 1), S, nb);
            if (minimum && onsph)
                p_ok("shear energy %.6e \u2192 %.6e (on sphere) \u2192 %.6e final (%.1f%%), %d iterations (%d rejected), %.2fs | minJ=%.3e  dev=%.2e",
                     E0, Es0, Efin, 100.0 * Efin / (Es0 > 0 ? Es0 : 1), it2, rejected, now() - t2, mj, radius_dev(S, nb));
            else
                p_warn("shear not at a minimum (iteration limit or sphere not reached): %.6e \u2192 %.6e, %d iterations (%d rejected), %.2fs | minJ=%.3e  dev=%.2e",
                       E0, Efin, it2, rejected, now() - t2, mj, radius_dev(S, nb));
        }
    }
    if (vid) { PCLOSE(vid); vid = NULL; }
    p_info("total %.2fs | CG %.2f (%ld its, %d GPU / %d CPU solves)", now() - tflow, tm.cg, tm.cg_its, tm.gpu_solves, tm.cpu_solves);
    if (has_tet) {
        nh.mode = 0; nh_set_ref(&nh, Vref0);
        double mj, mv; tet_min_stats(&nh, V, V, &mj, &mv);
        p_info("final minJ=%.3e  min tet vol=%.3e (start %.3e)  inverted=%s", mj, mv, init_min_vol, mj <= 0 ? "YES" : "0");
        char spath[1100];
        snprintf(spath, sizeof spath, "%s/stress_sphere.vtk", A->out);
        write_stress_fields(&nh, m, V, A->mu, A->kappa, spath);
        p_ok("pressure + deviatoric fields \u2192 %s", spath);
    }
    free(devh); free(S); free(U); free(mass); free(kfit); free(nrm); free(Vold); free(V0); free(Vref0);
}

/* ------------------------------------------------------------------------ */
/* CLI                                                                       */
/* ------------------------------------------------------------------------ */
static void usage(void)
{
    puts("usage: miso input.(stl|tet) [options]\n"
         "  -o DIR            output directory (default out)\n"
         "  --steps N         max flow steps (100)         --tol X     stop when radius deviation < X (1e-3)\n"
         "  --target area|volume\n"
         "  --delta-scale X   cMCF time step scale (1)     --min-cot X (1e-3)\n"
         "  --mu X --kappa X --fit X                        Neo-Hookean / boundary-fit stiffness (2, 5, 50)\n"
         "  --ref initial|updated                           reference configuration (initial)\n"
         "  --newton-iters N (15) --newton-tol X (1e-3) --max-ls N (30)\n"
         "  --cg-tol X (1e-2) --cg-iters N (400)            inexact Newton linear solve\n"
         "  --no-radial       never switch to radial projection\n"
         "  --radial-rate X   fraction of log-radius gap closed per step (0.3)\n"
         "  --star-every N    kernel-point search period while not star-shaped (5)\n"
         "  pipeline: boundary -> sphere (cMCF, radial projection once star-shaped), then the\n"
         "  interior is relaxed with off-diagonal Cauchy stress only until the shear energy is\n"
         "  minimal; outputs: volumetric_map.tet, flowed.vtk, stress_sphere.vtk\n"
         "  (pressure scalar + on-diagonal traceless deviatoric vector on the sphere).\n"
         "  --no-shear        stop after the boundary reaches the sphere\n"
         "  --shear-tol X     stop when relative shear-energy change < X (1e-4)   --shear-steps N (1000)\n"
         "  --fit-shear X     stiffness of the sphere penalty in phase 2 (2000)\n"
         "  --shear-tang X    tangential/normal penalty ratio: limits boundary sliding per iteration (0.01, adapted automatically)\n"
         "  --kappa-shear X   weight of the (ln J)^2 term that keeps J>0 in phase 2; keep small (0.1)\n"
         "  --video FILE      output video (default DIR/flow.mp4)   --no-video\n"
         "  --fps N (15)  --video-every N (1)  --video-size WxH (960x720)  --xvfb\n"
         "  --python CMD      interpreter for render_video.py (python3 / python on Windows)\n"
         "  --render-script P path of render_video.py (default: next to the executable)\n"
         "  --gpu auto|on|off OpenCL for the Newton CG solve (auto)\n"
         "  --gpu-min-rows N  auto: use GPU only if 3*nverts >= N (20000)\n"
         "  --cl-platform I --cl-device J   pick OpenCL device   --list-devices\n"
         "  --threads N       OpenMP threads\n"
         "  --allow-open     proceed despite an open / non-manifold / inconsistently wound STL");
}

static int parse_args(int argc, char **argv, Args *A)
{
    memset(A, 0, sizeof *A);
    A->out = "out"; A->steps = 100; A->tol = 1e-3; A->delta_scale = 1; A->min_cot = 1e-3;
    A->mu = 2; A->kappa = 5; A->fit = 50; A->newton_iters = 15; A->newton_tol = 1e-3; A->max_ls = 30;
    A->cg_tol = 1e-2; A->cg_iters = 400; A->radial = 1; A->radial_rate = 0.3; A->star_every = 5;
    A->shear = 1; A->shear_tol = 1e-4; A->shear_steps = 1000; A->fit_shear = 2000; A->shear_tang = 0.01; A->kappa_shear = 0.1; A->video_every = 1; A->fps = 15; A->vsize = "960x720"; A->gpu = 1; A->gpu_min_rows = 20000; A->cl_plat = A->cl_dev = -1;
#define IS(s) (!strcmp(argv[i], s))
#define NEXT (++i < argc ? argv[i] : (usage(), exit(1), (char *)0))
    for (int i = 1; i < argc; i++) {
        if (IS("-o") || IS("--out")) A->out = NEXT;
        else if (IS("--steps")) A->steps = atoi(NEXT);
        else if (IS("--tol")) A->tol = atof(NEXT);
        else if (IS("--target")) A->target_vol = !strcmp(NEXT, "volume");
        else if (IS("--delta-scale")) A->delta_scale = atof(NEXT);
        else if (IS("--min-cot")) A->min_cot = atof(NEXT);
        else if (IS("--mu")) A->mu = atof(NEXT);
        else if (IS("--kappa")) A->kappa = atof(NEXT);
        else if (IS("--fit")) A->fit = atof(NEXT);
        else if (IS("--ref")) A->ref_updated = !strcmp(NEXT, "updated");
        else if (IS("--newton-iters")) A->newton_iters = atoi(NEXT);
        else if (IS("--newton-tol")) A->newton_tol = atof(NEXT);
        else if (IS("--max-ls")) A->max_ls = atoi(NEXT);
        else if (IS("--cg-tol")) A->cg_tol = atof(NEXT);
        else if (IS("--cg-iters")) A->cg_iters = atoi(NEXT);
        else if (IS("--no-radial")) A->radial = 0;
        else if (IS("--radial-rate")) A->radial_rate = atof(NEXT);
        else if (IS("--star-every")) A->star_every = atoi(NEXT) > 0 ? atoi(argv[i]) : 1;
        else if (IS("--video")) A->video = NEXT;
        else if (IS("--no-video")) A->no_video = 1;
        else if (IS("--no-shear")) A->shear = 0;
        else if (IS("--shear-tol")) A->shear_tol = atof(NEXT);
        else if (IS("--shear-steps")) A->shear_steps = atoi(NEXT);
        else if (IS("--fit-shear")) A->fit_shear = atof(NEXT);
        else if (IS("--shear-tang")) A->shear_tang = atof(NEXT);
        else if (IS("--kappa-shear")) A->kappa_shear = atof(NEXT);
        else if (IS("--fps")) A->fps = atoi(NEXT);
        else if (IS("--video-every")) A->video_every = atoi(NEXT) > 0 ? atoi(argv[i]) : 1;
        else if (IS("--video-size")) A->vsize = NEXT;
        else if (IS("--xvfb")) A->xvfb = 1;
        else if (IS("--python")) A->python = NEXT;
        else if (IS("--render-script")) A->script = NEXT;
        else if (IS("--gpu")) { const char *v = NEXT; A->gpu = !strcmp(v, "off") ? 0 : !strcmp(v, "on") ? 2 : 1; }
        else if (IS("--gpu-min-rows")) A->gpu_min_rows = atoi(NEXT);
        else if (IS("--cl-platform")) A->cl_plat = atoi(NEXT);
        else if (IS("--cl-device")) A->cl_dev = atoi(NEXT);
        else if (IS("--list-devices")) A->list_dev = 1;
        else if (IS("--threads")) A->threads = atoi(NEXT);
        else if (IS("--allow-open")) A->allow_open = 1;
        else if (IS("-h") || IS("--help")) { usage(); exit(0); }
        else if (argv[i][0] != '-' && !A->input) A->input = argv[i];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); usage(); return 0; }
    }
    if (A->radial_rate <= 0 || A->radial_rate > 1) A->radial_rate = 0.3;
    if (!A->python) {
#ifdef _WIN32
        A->python = "python";
#else
        A->python = "python3";
#endif
    }
    return A->input != NULL || A->list_dev;
}

static int has_ext(const char *p, const char *ext)
{
    size_t l = strlen(p), e = strlen(ext);
    if (l < e) return 0;
    for (size_t i = 0; i < e; i++) if (tolower((unsigned char)p[l - e + i]) != ext[i]) return 0;
    return 1;
}

int main(int argc, char **argv)
{
    Args A;
    term_init();
    banner_print();
    if (!parse_args(argc, argv, &A)) { usage(); return 1; }
    {
        snprintf(g_exe_dir, sizeof g_exe_dir, "%s", argv[0]);
        char *sl = strrchr(g_exe_dir, '/'), *bs = strrchr(g_exe_dir, '\\');
        if (bs > sl) sl = bs;
        if (sl) *sl = 0; else strcpy(g_exe_dir, ".");
    }
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif
    char vpath[1100];
    if (!A.video) { snprintf(vpath, sizeof vpath, "%s/flow.mp4", A.out); A.video = vpath; }
#ifdef _OPENMP
    if (A.threads > 0) omp_set_num_threads(A.threads);
#endif
    if (A.list_dev) {
#ifdef USE_OPENCL
        gpu_list();
#else
        puts("built without OpenCL (recompile with -DUSE_OPENCL -lOpenCL)");
#endif
        return 0;
    }
    MKDIR(A.out);

    Mesh m; memset(&m, 0, sizeof m);
    if (has_ext(A.input, ".tet")) {
        if (!load_tet(A.input, &m)) return 1;
        boundary_from_tets(&m);
    } else if (has_ext(A.input, ".stl")) {
        if (!load_stl(A.input, &m, A.allow_open)) return 1;
    } else { p_err("input must be .stl (surface only) or .tet"); return 1; }
    build_boundary_index(&m);
    p_info("%d verts, %d tets, %d boundary verts, %d boundary faces%s", m.nv, m.nt, m.nb, m.nf,
           m.T ? "" : "  (surface-only)");
#ifdef _OPENMP
    p_info("OpenMP threads: %d", omp_get_max_threads());
#endif

    GPU gpu; memset(&gpu, 0, sizeof gpu);
#ifdef USE_OPENCL
    if (m.T && A.gpu) {
        if (!gpu_init(&gpu, A.cl_plat, A.cl_dev, A.gpu == 2)) p_info("no usable OpenCL device, CPU only");
    }
#else
    if (m.T && A.gpu == 2) p_warn("--gpu on requested but built without OpenCL");
#endif

    double *V0 = xmalloc(sizeof(double) * 3 * (size_t)m.nv), *V = xmalloc(sizeof(double) * 3 * (size_t)m.nv);
    (void)V0;
    memcpy(V, m.V, sizeof(double) * 3 * (size_t)m.nv);
    flow(&m, V, &A, &gpu);

    char p[1024];
    snprintf(p, sizeof p, "%s/flowed.vtk", A.out); write_vtk(&m, V, p);
    snprintf(p, sizeof p, "%s/boundary_flowed.stl", A.out); write_stl(&m, V, p);
    if (m.T) { snprintf(p, sizeof p, "%s/volumetric_map.tet", A.out); write_tet(&m, V, p); }
    p_ok("outputs in %s", A.out);
    return 0;
}

	.file	"blit.ispc"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function blit_rgba8888___un_3C_unu_3E_un_3C_Cunu_3E_uniuniuniuniuniuni
.LCPI0_0:
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
.LCPI0_1:
	.long	255                             # 0xff
.LCPI0_2:
	.long	128                             # 0x80
.LCPI0_3:
	.long	1                               # 0x1
.LCPI0_4:
	.long	0x3f800000                      # float 1
.LCPI0_7:
	.long	32768                           # 0x8000
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0
.LCPI0_5:
	.byte	1                               # 0x1
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	5                               # 0x5
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	9                               # 0x9
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	13                              # 0xd
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	17                              # 0x11
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	21                              # 0x15
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	25                              # 0x19
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	29                              # 0x1d
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
.LCPI0_6:
	.byte	2                               # 0x2
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	6                               # 0x6
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	10                              # 0xa
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	14                              # 0xe
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	18                              # 0x12
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	22                              # 0x16
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	26                              # 0x1a
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	30                              # 0x1e
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
.LCPI0_8:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
	.long	4                               # 0x4
	.long	5                               # 0x5
	.long	6                               # 0x6
	.long	7                               # 0x7
	.text
	.globl	blit_rgba8888___un_3C_unu_3E_un_3C_Cunu_3E_uniuniuniuniuniuni
	.p2align	4
	.type	blit_rgba8888___un_3C_unu_3E_un_3C_Cunu_3E_uniuniuniuniuniuni,@function
blit_rgba8888___un_3C_unu_3E_un_3C_Cunu_3E_uniuniuniuniuniuni: # @blit_rgba8888___un_3C_unu_3E_un_3C_Cunu_3E_uniuniuniuniuniuni
# %bb.0:
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
                                        # kill: def $r8d killed $r8d def $r8
	movl	48(%rsp), %r10d
	movl	40(%rsp), %r11d
	vpcmpeqd	%ymm1, %ymm1, %ymm1
	cmpl	$255, %r11d
	movl	$256, %eax                      # imm = 0x100
	cmovel	%eax, %r11d
	cmpl	$255, %r10d
	cmovel	%eax, %r10d
	vtestps	%ymm1, %ymm0
	jb	.LBB0_9
# %bb.1:
	testl	%r9d, %r9d
	jle	.LBB0_17
# %bb.2:
	leal	7(%r8), %eax
	testl	%r8d, %r8d
	cmovnsl	%r8d, %eax
	andl	$-8, %eax
	vmovd	%r11d, %xmm0
	vpbroadcastd	%xmm0, %ymm7
	vmovd	%r10d, %xmm1
	vpbroadcastd	%xmm1, %ymm11
	vmovd	%r8d, %xmm2
	vpbroadcastd	%xmm2, %ymm0
	vmovdqu	%ymm0, -72(%rsp)                # 32-byte Spill
	movslq	%eax, %r10
	xorl	%r11d, %r11d
	vpbroadcastd	.LCPI0_0(%rip), %ymm3   # ymm3 = [0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255]
	vpbroadcastd	.LCPI0_1(%rip), %ymm4   # ymm4 = [255,255,255,255,255,255,255,255]
	vpbroadcastd	.LCPI0_2(%rip), %ymm5   # ymm5 = [128,128,128,128,128,128,128,128]
	vbroadcastss	.LCPI0_3(%rip), %ymm8   # ymm8 = [1,1,1,1,1,1,1,1]
	vbroadcastss	.LCPI0_4(%rip), %ymm9   # ymm9 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vmovdqa	.LCPI0_5(%rip), %ymm10          # ymm10 = [1,128,128,128,5,128,128,128,9,128,128,128,13,128,128,128,17,128,128,128,21,128,128,128,25,128,128,128,29,128,128,128]
	vpbroadcastd	.LCPI0_7(%rip), %ymm1   # ymm1 = [32768,32768,32768,32768,32768,32768,32768,32768]
	vmovdqu	%ymm7, -104(%rsp)               # 32-byte Spill
	vmovdqa	%ymm11, %ymm6
	jmp	.LBB0_3
	.p2align	4
.LBB0_8:                                #   in Loop: Header=BB0_3 Depth=1
	incl	%r11d
	cmpl	%r9d, %r11d
	je	.LBB0_17
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
	movl	%r11d, %ebx
	imull	%edx, %ebx
	sarl	$2, %ebx
	movl	%r11d, %ebp
	imull	%ecx, %ebp
	sarl	$2, %ebp
	movslq	%ebx, %rbx
	leaq	(%rdi,%rbx,4), %rbx
	movslq	%ebp, %r14
	leaq	(%rsi,%r14,4), %r14
	movl	$0, %r15d
	testl	%eax, %eax
	jle	.LBB0_6
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=1
	xorl	%r15d, %r15d
	.p2align	4
.LBB0_5:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovdqu	(%rbx,%r15,4), %ymm13
	vmovdqu	(%r14,%r15,4), %ymm14
	vpsrld	$24, %ymm13, %ymm2
	vpmulld	%ymm7, %ymm2, %ymm2
	vpsrld	$8, %ymm2, %ymm2
	vpsrld	$24, %ymm14, %ymm12
	vpmulld	%ymm11, %ymm12, %ymm12
	vpsrld	$8, %ymm12, %ymm7
	vpand	%ymm3, %ymm12, %ymm15
	vpsubd	%ymm7, %ymm4, %ymm7
	vpmulld	%ymm2, %ymm7, %ymm12
	vpaddd	%ymm5, %ymm15, %ymm2
	vpaddd	%ymm2, %ymm12, %ymm2
	vpsrad	$8, %ymm2, %ymm7
	vpslld	$24, %ymm7, %ymm11
	vpminud	%ymm4, %ymm2, %ymm0
	vpcmpeqd	%ymm0, %ymm2, %ymm0
	vblendvps	%ymm0, %ymm8, %ymm7, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vdivps	%ymm0, %ymm9, %ymm2
	vpand	%ymm4, %ymm13, %ymm0
	vpand	%ymm4, %ymm14, %ymm7
	vpmulld	%ymm7, %ymm15, %ymm7
	vpmulld	%ymm0, %ymm12, %ymm0
	vpaddd	%ymm7, %ymm0, %ymm0
	vpshufb	%ymm10, %ymm14, %ymm7
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm2, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm5, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpor	%ymm0, %ymm11, %ymm0
	vpmulld	%ymm7, %ymm15, %ymm7
	vpshufb	%ymm10, %ymm13, %ymm11
	vpmulld	%ymm11, %ymm12, %ymm11
	vpaddd	%ymm7, %ymm11, %ymm7
	vpshufb	.LCPI0_6(%rip), %ymm14, %ymm11  # ymm11 = ymm14[2],zero,zero,zero,ymm14[6],zero,zero,zero,ymm14[10],zero,zero,zero,ymm14[14],zero,zero,zero,ymm14[18],zero,zero,zero,ymm14[22],zero,zero,zero,ymm14[26],zero,zero,zero,ymm14[30],zero,zero,zero
	vpmulld	%ymm11, %ymm15, %ymm11
	vpshufb	.LCPI0_6(%rip), %ymm13, %ymm13  # ymm13 = ymm13[2],zero,zero,zero,ymm13[6],zero,zero,zero,ymm13[10],zero,zero,zero,ymm13[14],zero,zero,zero,ymm13[18],zero,zero,zero,ymm13[22],zero,zero,zero,ymm13[26],zero,zero,zero,ymm13[30],zero,zero,zero
	vpmulld	%ymm13, %ymm12, %ymm12
	vpaddd	%ymm11, %ymm12, %ymm11
	vcvtdq2ps	%ymm7, %ymm7
	vmulps	%ymm7, %ymm2, %ymm7
	vcvtdq2ps	%ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm2
	vmovdqa	%ymm6, %ymm11
	vcvttps2dq	%ymm7, %ymm7
	vpaddd	%ymm5, %ymm7, %ymm7
	vpand	%ymm3, %ymm7, %ymm7
	vpor	%ymm7, %ymm0, %ymm0
	vmovdqu	-104(%rsp), %ymm7               # 32-byte Reload
	vcvttps2dq	%ymm2, %ymm2
	vpslld	$8, %ymm2, %ymm2
	vpaddd	%ymm1, %ymm2, %ymm2
	vpxor	%xmm12, %xmm12, %xmm12
	vpblendw	$170, %ymm2, %ymm12, %ymm2      # ymm2 = ymm12[0],ymm2[1],ymm12[2],ymm2[3],ymm12[4],ymm2[5],ymm12[6],ymm2[7],ymm12[8],ymm2[9],ymm12[10],ymm2[11],ymm12[12],ymm2[13],ymm12[14],ymm2[15]
	vpor	%ymm2, %ymm0, %ymm0
	vmovdqu	%ymm0, (%rbx,%r15,4)
	addq	$8, %r15
	cmpq	%r10, %r15
	jl	.LBB0_5
.LBB0_6:                                #   in Loop: Header=BB0_3 Depth=1
	cmpl	%r8d, %r15d
	jge	.LBB0_8
# %bb.7:                                #   in Loop: Header=BB0_3 Depth=1
	vmovd	%r15d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vpor	.LCPI0_8(%rip), %ymm0, %ymm0
	vmovdqu	-72(%rsp), %ymm2                # 32-byte Reload
	vpcmpgtd	%ymm0, %ymm2, %ymm0
	vmovdqu	%ymm0, -40(%rsp)                # 32-byte Spill
	movl	%r15d, %r15d
	vmaskmovps	(%rbx,%r15,4), %ymm0, %ymm14
	vmaskmovps	(%r14,%r15,4), %ymm0, %ymm15
	vpsrld	$24, %ymm14, %ymm0
	vpmulld	%ymm7, %ymm0, %ymm0
	vpsrld	$8, %ymm0, %ymm0
	vpsrld	$24, %ymm15, %ymm2
	vpmulld	%ymm11, %ymm2, %ymm2
	vpsrld	$8, %ymm2, %ymm7
	vpand	%ymm3, %ymm2, %ymm12
	vpsubd	%ymm7, %ymm4, %ymm2
	vpmulld	%ymm0, %ymm2, %ymm2
	vpaddd	%ymm5, %ymm12, %ymm0
	vpaddd	%ymm0, %ymm2, %ymm0
	vpsrad	$8, %ymm0, %ymm7
	vpslld	$24, %ymm7, %ymm11
	vpminud	%ymm4, %ymm0, %ymm13
	vpcmpeqd	%ymm0, %ymm13, %ymm0
	vblendvps	%ymm0, %ymm8, %ymm7, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vdivps	%ymm0, %ymm9, %ymm13
	vpand	%ymm4, %ymm14, %ymm0
	vpand	%ymm4, %ymm15, %ymm7
	vpmulld	%ymm7, %ymm12, %ymm7
	vpmulld	%ymm0, %ymm2, %ymm0
	vpaddd	%ymm7, %ymm0, %ymm0
	vpshufb	%ymm10, %ymm15, %ymm7
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm13, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm5, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpor	%ymm0, %ymm11, %ymm0
	vpmulld	%ymm7, %ymm12, %ymm7
	vpshufb	%ymm10, %ymm14, %ymm11
	vpmulld	%ymm11, %ymm2, %ymm11
	vpaddd	%ymm7, %ymm11, %ymm7
	vpshufb	.LCPI0_6(%rip), %ymm15, %ymm11  # ymm11 = ymm15[2],zero,zero,zero,ymm15[6],zero,zero,zero,ymm15[10],zero,zero,zero,ymm15[14],zero,zero,zero,ymm15[18],zero,zero,zero,ymm15[22],zero,zero,zero,ymm15[26],zero,zero,zero,ymm15[30],zero,zero,zero
	vpmulld	%ymm11, %ymm12, %ymm11
	vpshufb	.LCPI0_6(%rip), %ymm14, %ymm12  # ymm12 = ymm14[2],zero,zero,zero,ymm14[6],zero,zero,zero,ymm14[10],zero,zero,zero,ymm14[14],zero,zero,zero,ymm14[18],zero,zero,zero,ymm14[22],zero,zero,zero,ymm14[26],zero,zero,zero,ymm14[30],zero,zero,zero
	vpmulld	%ymm12, %ymm2, %ymm2
	vpaddd	%ymm2, %ymm11, %ymm2
	vmovdqa	%ymm6, %ymm11
	vcvtdq2ps	%ymm7, %ymm7
	vmulps	%ymm7, %ymm13, %ymm7
	vcvtdq2ps	%ymm2, %ymm2
	vmulps	%ymm2, %ymm13, %ymm2
	vcvttps2dq	%ymm7, %ymm7
	vpaddd	%ymm5, %ymm7, %ymm7
	vpand	%ymm3, %ymm7, %ymm7
	vpor	%ymm7, %ymm0, %ymm0
	vmovdqu	-104(%rsp), %ymm7               # 32-byte Reload
	vcvttps2dq	%ymm2, %ymm2
	vpslld	$8, %ymm2, %ymm2
	vpaddd	%ymm1, %ymm2, %ymm2
	vpxor	%xmm12, %xmm12, %xmm12
	vpblendw	$170, %ymm2, %ymm12, %ymm2      # ymm2 = ymm12[0],ymm2[1],ymm12[2],ymm2[3],ymm12[4],ymm2[5],ymm12[6],ymm2[7],ymm12[8],ymm2[9],ymm12[10],ymm2[11],ymm12[12],ymm2[13],ymm12[14],ymm2[15]
	vpor	%ymm2, %ymm0, %ymm0
	vmovups	-40(%rsp), %ymm2                # 32-byte Reload
	vmaskmovps	%ymm0, %ymm2, (%rbx,%r15,4)
	jmp	.LBB0_8
.LBB0_9:
	testl	%r9d, %r9d
	jle	.LBB0_17
# %bb.10:
	leal	7(%r8), %eax
	testl	%r8d, %r8d
	cmovnsl	%r8d, %eax
	andl	$-8, %eax
	vmovd	%r11d, %xmm0
	vpbroadcastd	%xmm0, %ymm7
	vmovd	%r10d, %xmm1
	vpbroadcastd	%xmm1, %ymm11
	vmovd	%r8d, %xmm2
	vpbroadcastd	%xmm2, %ymm0
	vmovdqu	%ymm0, -72(%rsp)                # 32-byte Spill
	movslq	%eax, %r10
	xorl	%r11d, %r11d
	vpbroadcastd	.LCPI0_0(%rip), %ymm3   # ymm3 = [0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255]
	vpbroadcastd	.LCPI0_1(%rip), %ymm4   # ymm4 = [255,255,255,255,255,255,255,255]
	vpbroadcastd	.LCPI0_2(%rip), %ymm5   # ymm5 = [128,128,128,128,128,128,128,128]
	vbroadcastss	.LCPI0_3(%rip), %ymm8   # ymm8 = [1,1,1,1,1,1,1,1]
	vbroadcastss	.LCPI0_4(%rip), %ymm9   # ymm9 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vmovdqa	.LCPI0_5(%rip), %ymm10          # ymm10 = [1,128,128,128,5,128,128,128,9,128,128,128,13,128,128,128,17,128,128,128,21,128,128,128,25,128,128,128,29,128,128,128]
	vpbroadcastd	.LCPI0_7(%rip), %ymm1   # ymm1 = [32768,32768,32768,32768,32768,32768,32768,32768]
	vmovdqu	%ymm7, -104(%rsp)               # 32-byte Spill
	vmovdqa	%ymm11, %ymm6
	jmp	.LBB0_11
	.p2align	4
.LBB0_16:                               #   in Loop: Header=BB0_11 Depth=1
	incl	%r11d
	cmpl	%r9d, %r11d
	je	.LBB0_17
.LBB0_11:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_13 Depth 2
	movl	%r11d, %ebx
	imull	%edx, %ebx
	sarl	$2, %ebx
	movl	%r11d, %ebp
	imull	%ecx, %ebp
	sarl	$2, %ebp
	movslq	%ebx, %rbx
	leaq	(%rdi,%rbx,4), %rbx
	movslq	%ebp, %r14
	leaq	(%rsi,%r14,4), %r14
	movl	$0, %r15d
	testl	%eax, %eax
	jle	.LBB0_14
# %bb.12:                               #   in Loop: Header=BB0_11 Depth=1
	xorl	%r15d, %r15d
	.p2align	4
.LBB0_13:                               #   Parent Loop BB0_11 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovdqu	(%rbx,%r15,4), %ymm13
	vmovdqu	(%r14,%r15,4), %ymm14
	vpsrld	$24, %ymm13, %ymm2
	vpmulld	%ymm7, %ymm2, %ymm2
	vpsrld	$8, %ymm2, %ymm2
	vpsrld	$24, %ymm14, %ymm12
	vpmulld	%ymm11, %ymm12, %ymm12
	vpsrld	$8, %ymm12, %ymm7
	vpand	%ymm3, %ymm12, %ymm15
	vpsubd	%ymm7, %ymm4, %ymm7
	vpmulld	%ymm2, %ymm7, %ymm12
	vpaddd	%ymm5, %ymm15, %ymm2
	vpaddd	%ymm2, %ymm12, %ymm2
	vpsrad	$8, %ymm2, %ymm7
	vpslld	$24, %ymm7, %ymm11
	vpminud	%ymm4, %ymm2, %ymm0
	vpcmpeqd	%ymm0, %ymm2, %ymm0
	vblendvps	%ymm0, %ymm8, %ymm7, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vdivps	%ymm0, %ymm9, %ymm2
	vpand	%ymm4, %ymm13, %ymm0
	vpand	%ymm4, %ymm14, %ymm7
	vpmulld	%ymm7, %ymm15, %ymm7
	vpmulld	%ymm0, %ymm12, %ymm0
	vpaddd	%ymm7, %ymm0, %ymm0
	vpshufb	%ymm10, %ymm14, %ymm7
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm2, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm5, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpor	%ymm0, %ymm11, %ymm0
	vpmulld	%ymm7, %ymm15, %ymm7
	vpshufb	%ymm10, %ymm13, %ymm11
	vpmulld	%ymm11, %ymm12, %ymm11
	vpaddd	%ymm7, %ymm11, %ymm7
	vpshufb	.LCPI0_6(%rip), %ymm14, %ymm11  # ymm11 = ymm14[2],zero,zero,zero,ymm14[6],zero,zero,zero,ymm14[10],zero,zero,zero,ymm14[14],zero,zero,zero,ymm14[18],zero,zero,zero,ymm14[22],zero,zero,zero,ymm14[26],zero,zero,zero,ymm14[30],zero,zero,zero
	vpmulld	%ymm11, %ymm15, %ymm11
	vpshufb	.LCPI0_6(%rip), %ymm13, %ymm13  # ymm13 = ymm13[2],zero,zero,zero,ymm13[6],zero,zero,zero,ymm13[10],zero,zero,zero,ymm13[14],zero,zero,zero,ymm13[18],zero,zero,zero,ymm13[22],zero,zero,zero,ymm13[26],zero,zero,zero,ymm13[30],zero,zero,zero
	vpmulld	%ymm13, %ymm12, %ymm12
	vpaddd	%ymm11, %ymm12, %ymm11
	vcvtdq2ps	%ymm7, %ymm7
	vmulps	%ymm7, %ymm2, %ymm7
	vcvtdq2ps	%ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm2
	vmovdqa	%ymm6, %ymm11
	vcvttps2dq	%ymm7, %ymm7
	vpaddd	%ymm5, %ymm7, %ymm7
	vpand	%ymm3, %ymm7, %ymm7
	vpor	%ymm7, %ymm0, %ymm0
	vmovdqu	-104(%rsp), %ymm7               # 32-byte Reload
	vcvttps2dq	%ymm2, %ymm2
	vpslld	$8, %ymm2, %ymm2
	vpaddd	%ymm1, %ymm2, %ymm2
	vpxor	%xmm12, %xmm12, %xmm12
	vpblendw	$170, %ymm2, %ymm12, %ymm2      # ymm2 = ymm12[0],ymm2[1],ymm12[2],ymm2[3],ymm12[4],ymm2[5],ymm12[6],ymm2[7],ymm12[8],ymm2[9],ymm12[10],ymm2[11],ymm12[12],ymm2[13],ymm12[14],ymm2[15]
	vpor	%ymm2, %ymm0, %ymm0
	vmovdqu	%ymm0, (%rbx,%r15,4)
	addq	$8, %r15
	cmpq	%r10, %r15
	jl	.LBB0_13
.LBB0_14:                               #   in Loop: Header=BB0_11 Depth=1
	cmpl	%r8d, %r15d
	jge	.LBB0_16
# %bb.15:                               #   in Loop: Header=BB0_11 Depth=1
	vmovd	%r15d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vpor	.LCPI0_8(%rip), %ymm0, %ymm0
	vmovdqu	-72(%rsp), %ymm2                # 32-byte Reload
	vpcmpgtd	%ymm0, %ymm2, %ymm0
	vmovdqu	%ymm0, -40(%rsp)                # 32-byte Spill
	movl	%r15d, %r15d
	vmaskmovps	(%rbx,%r15,4), %ymm0, %ymm14
	vmaskmovps	(%r14,%r15,4), %ymm0, %ymm15
	vpsrld	$24, %ymm14, %ymm0
	vpmulld	%ymm7, %ymm0, %ymm0
	vpsrld	$8, %ymm0, %ymm0
	vpsrld	$24, %ymm15, %ymm2
	vpmulld	%ymm11, %ymm2, %ymm2
	vpsrld	$8, %ymm2, %ymm7
	vpand	%ymm3, %ymm2, %ymm12
	vpsubd	%ymm7, %ymm4, %ymm2
	vpmulld	%ymm0, %ymm2, %ymm2
	vpaddd	%ymm5, %ymm12, %ymm0
	vpaddd	%ymm0, %ymm2, %ymm0
	vpsrad	$8, %ymm0, %ymm7
	vpslld	$24, %ymm7, %ymm11
	vpminud	%ymm4, %ymm0, %ymm13
	vpcmpeqd	%ymm0, %ymm13, %ymm0
	vblendvps	%ymm0, %ymm8, %ymm7, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vdivps	%ymm0, %ymm9, %ymm13
	vpand	%ymm4, %ymm14, %ymm0
	vpand	%ymm4, %ymm15, %ymm7
	vpmulld	%ymm7, %ymm12, %ymm7
	vpmulld	%ymm0, %ymm2, %ymm0
	vpaddd	%ymm7, %ymm0, %ymm0
	vpshufb	%ymm10, %ymm15, %ymm7
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm13, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm5, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpor	%ymm0, %ymm11, %ymm0
	vpmulld	%ymm7, %ymm12, %ymm7
	vpshufb	%ymm10, %ymm14, %ymm11
	vpmulld	%ymm11, %ymm2, %ymm11
	vpaddd	%ymm7, %ymm11, %ymm7
	vpshufb	.LCPI0_6(%rip), %ymm15, %ymm11  # ymm11 = ymm15[2],zero,zero,zero,ymm15[6],zero,zero,zero,ymm15[10],zero,zero,zero,ymm15[14],zero,zero,zero,ymm15[18],zero,zero,zero,ymm15[22],zero,zero,zero,ymm15[26],zero,zero,zero,ymm15[30],zero,zero,zero
	vpmulld	%ymm11, %ymm12, %ymm11
	vpshufb	.LCPI0_6(%rip), %ymm14, %ymm12  # ymm12 = ymm14[2],zero,zero,zero,ymm14[6],zero,zero,zero,ymm14[10],zero,zero,zero,ymm14[14],zero,zero,zero,ymm14[18],zero,zero,zero,ymm14[22],zero,zero,zero,ymm14[26],zero,zero,zero,ymm14[30],zero,zero,zero
	vpmulld	%ymm12, %ymm2, %ymm2
	vpaddd	%ymm2, %ymm11, %ymm2
	vmovdqa	%ymm6, %ymm11
	vcvtdq2ps	%ymm7, %ymm7
	vmulps	%ymm7, %ymm13, %ymm7
	vcvtdq2ps	%ymm2, %ymm2
	vmulps	%ymm2, %ymm13, %ymm2
	vcvttps2dq	%ymm7, %ymm7
	vpaddd	%ymm5, %ymm7, %ymm7
	vpand	%ymm3, %ymm7, %ymm7
	vpor	%ymm7, %ymm0, %ymm0
	vmovdqu	-104(%rsp), %ymm7               # 32-byte Reload
	vcvttps2dq	%ymm2, %ymm2
	vpslld	$8, %ymm2, %ymm2
	vpaddd	%ymm1, %ymm2, %ymm2
	vpxor	%xmm12, %xmm12, %xmm12
	vpblendw	$170, %ymm2, %ymm12, %ymm2      # ymm2 = ymm12[0],ymm2[1],ymm12[2],ymm2[3],ymm12[4],ymm2[5],ymm12[6],ymm2[7],ymm12[8],ymm2[9],ymm12[10],ymm2[11],ymm12[12],ymm2[13],ymm12[14],ymm2[15]
	vpor	%ymm2, %ymm0, %ymm0
	vmovups	-40(%rsp), %ymm2                # 32-byte Reload
	vmaskmovps	%ymm0, %ymm2, (%rbx,%r15,4)
	jmp	.LBB0_16
.LBB0_17:
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end0:
	.size	blit_rgba8888___un_3C_unu_3E_un_3C_Cunu_3E_uniuniuniuniuniuni, .Lfunc_end0-blit_rgba8888___un_3C_unu_3E_un_3C_Cunu_3E_uniuniuniuniuniuni
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function blit_rgba8888
.LCPI1_0:
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
.LCPI1_1:
	.long	255                             # 0xff
.LCPI1_2:
	.long	128                             # 0x80
.LCPI1_3:
	.long	1                               # 0x1
.LCPI1_4:
	.long	0x3f800000                      # float 1
.LCPI1_7:
	.long	32768                           # 0x8000
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0
.LCPI1_5:
	.byte	1                               # 0x1
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	5                               # 0x5
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	9                               # 0x9
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	13                              # 0xd
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	17                              # 0x11
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	21                              # 0x15
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	25                              # 0x19
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	29                              # 0x1d
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
.LCPI1_6:
	.byte	2                               # 0x2
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	6                               # 0x6
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	10                              # 0xa
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	14                              # 0xe
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	18                              # 0x12
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	22                              # 0x16
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	26                              # 0x1a
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	30                              # 0x1e
	.byte	128                             # 0x80
	.byte	128                             # 0x80
	.byte	128                             # 0x80
.LCPI1_8:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
	.long	4                               # 0x4
	.long	5                               # 0x5
	.long	6                               # 0x6
	.long	7                               # 0x7
	.text
	.globl	blit_rgba8888
	.p2align	4
	.type	blit_rgba8888,@function
blit_rgba8888:                          # @blit_rgba8888
# %bb.0:
                                        # kill: def $r8d killed $r8d def $r8
	testl	%r9d, %r9d
	jle	.LBB1_9
# %bb.1:
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	movl	48(%rsp), %r10d
	cmpl	$255, %r10d
	movl	$256, %eax                      # imm = 0x100
	cmovel	%eax, %r10d
	movl	40(%rsp), %r11d
	cmpl	$255, %r11d
	cmovel	%eax, %r11d
	leal	7(%r8), %eax
	testl	%r8d, %r8d
	cmovnsl	%r8d, %eax
	andl	$-8, %eax
	vmovd	%r11d, %xmm0
	vpbroadcastd	%xmm0, %ymm7
	vmovd	%r10d, %xmm1
	vpbroadcastd	%xmm1, %ymm11
	vmovd	%r8d, %xmm2
	vpbroadcastd	%xmm2, %ymm0
	vmovdqu	%ymm0, -72(%rsp)                # 32-byte Spill
	movslq	%eax, %r10
	xorl	%r11d, %r11d
	vpbroadcastd	.LCPI1_0(%rip), %ymm3   # ymm3 = [0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255]
	vpbroadcastd	.LCPI1_1(%rip), %ymm4   # ymm4 = [255,255,255,255,255,255,255,255]
	vpbroadcastd	.LCPI1_2(%rip), %ymm5   # ymm5 = [128,128,128,128,128,128,128,128]
	vbroadcastss	.LCPI1_3(%rip), %ymm8   # ymm8 = [1,1,1,1,1,1,1,1]
	vbroadcastss	.LCPI1_4(%rip), %ymm9   # ymm9 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vmovdqa	.LCPI1_5(%rip), %ymm10          # ymm10 = [1,128,128,128,5,128,128,128,9,128,128,128,13,128,128,128,17,128,128,128,21,128,128,128,25,128,128,128,29,128,128,128]
	vpbroadcastd	.LCPI1_7(%rip), %ymm1   # ymm1 = [32768,32768,32768,32768,32768,32768,32768,32768]
	vmovdqu	%ymm7, -104(%rsp)               # 32-byte Spill
	vmovdqa	%ymm11, %ymm6
	jmp	.LBB1_2
	.p2align	4
.LBB1_7:                                #   in Loop: Header=BB1_2 Depth=1
	incl	%r11d
	cmpl	%r9d, %r11d
	je	.LBB1_8
.LBB1_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_4 Depth 2
	movl	%r11d, %ebx
	imull	%edx, %ebx
	sarl	$2, %ebx
	movl	%r11d, %ebp
	imull	%ecx, %ebp
	sarl	$2, %ebp
	movslq	%ebx, %rbx
	leaq	(%rdi,%rbx,4), %rbx
	movslq	%ebp, %r14
	leaq	(%rsi,%r14,4), %r14
	movl	$0, %r15d
	testl	%eax, %eax
	jle	.LBB1_5
# %bb.3:                                #   in Loop: Header=BB1_2 Depth=1
	xorl	%r15d, %r15d
	.p2align	4
.LBB1_4:                                #   Parent Loop BB1_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovdqu	(%rbx,%r15,4), %ymm13
	vmovdqu	(%r14,%r15,4), %ymm14
	vpsrld	$24, %ymm13, %ymm2
	vpmulld	%ymm7, %ymm2, %ymm2
	vpsrld	$8, %ymm2, %ymm2
	vpsrld	$24, %ymm14, %ymm12
	vpmulld	%ymm11, %ymm12, %ymm12
	vpsrld	$8, %ymm12, %ymm7
	vpand	%ymm3, %ymm12, %ymm15
	vpsubd	%ymm7, %ymm4, %ymm7
	vpmulld	%ymm2, %ymm7, %ymm12
	vpaddd	%ymm5, %ymm15, %ymm2
	vpaddd	%ymm2, %ymm12, %ymm2
	vpsrad	$8, %ymm2, %ymm7
	vpslld	$24, %ymm7, %ymm11
	vpminud	%ymm4, %ymm2, %ymm0
	vpcmpeqd	%ymm0, %ymm2, %ymm0
	vblendvps	%ymm0, %ymm8, %ymm7, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vdivps	%ymm0, %ymm9, %ymm2
	vpand	%ymm4, %ymm13, %ymm0
	vpand	%ymm4, %ymm14, %ymm7
	vpmulld	%ymm7, %ymm15, %ymm7
	vpmulld	%ymm0, %ymm12, %ymm0
	vpaddd	%ymm7, %ymm0, %ymm0
	vpshufb	%ymm10, %ymm14, %ymm7
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm2, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm5, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpor	%ymm0, %ymm11, %ymm0
	vpmulld	%ymm7, %ymm15, %ymm7
	vpshufb	%ymm10, %ymm13, %ymm11
	vpmulld	%ymm11, %ymm12, %ymm11
	vpaddd	%ymm7, %ymm11, %ymm7
	vpshufb	.LCPI1_6(%rip), %ymm14, %ymm11  # ymm11 = ymm14[2],zero,zero,zero,ymm14[6],zero,zero,zero,ymm14[10],zero,zero,zero,ymm14[14],zero,zero,zero,ymm14[18],zero,zero,zero,ymm14[22],zero,zero,zero,ymm14[26],zero,zero,zero,ymm14[30],zero,zero,zero
	vpmulld	%ymm11, %ymm15, %ymm11
	vpshufb	.LCPI1_6(%rip), %ymm13, %ymm13  # ymm13 = ymm13[2],zero,zero,zero,ymm13[6],zero,zero,zero,ymm13[10],zero,zero,zero,ymm13[14],zero,zero,zero,ymm13[18],zero,zero,zero,ymm13[22],zero,zero,zero,ymm13[26],zero,zero,zero,ymm13[30],zero,zero,zero
	vpmulld	%ymm13, %ymm12, %ymm12
	vpaddd	%ymm11, %ymm12, %ymm11
	vcvtdq2ps	%ymm7, %ymm7
	vmulps	%ymm7, %ymm2, %ymm7
	vcvtdq2ps	%ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm2
	vmovdqa	%ymm6, %ymm11
	vcvttps2dq	%ymm7, %ymm7
	vpaddd	%ymm5, %ymm7, %ymm7
	vpand	%ymm3, %ymm7, %ymm7
	vpor	%ymm7, %ymm0, %ymm0
	vmovdqu	-104(%rsp), %ymm7               # 32-byte Reload
	vcvttps2dq	%ymm2, %ymm2
	vpslld	$8, %ymm2, %ymm2
	vpaddd	%ymm1, %ymm2, %ymm2
	vpxor	%xmm12, %xmm12, %xmm12
	vpblendw	$170, %ymm2, %ymm12, %ymm2      # ymm2 = ymm12[0],ymm2[1],ymm12[2],ymm2[3],ymm12[4],ymm2[5],ymm12[6],ymm2[7],ymm12[8],ymm2[9],ymm12[10],ymm2[11],ymm12[12],ymm2[13],ymm12[14],ymm2[15]
	vpor	%ymm2, %ymm0, %ymm0
	vmovdqu	%ymm0, (%rbx,%r15,4)
	addq	$8, %r15
	cmpq	%r10, %r15
	jl	.LBB1_4
.LBB1_5:                                #   in Loop: Header=BB1_2 Depth=1
	cmpl	%r8d, %r15d
	jge	.LBB1_7
# %bb.6:                                #   in Loop: Header=BB1_2 Depth=1
	vmovd	%r15d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vpor	.LCPI1_8(%rip), %ymm0, %ymm0
	vmovdqu	-72(%rsp), %ymm2                # 32-byte Reload
	vpcmpgtd	%ymm0, %ymm2, %ymm0
	vmovdqu	%ymm0, -40(%rsp)                # 32-byte Spill
	movl	%r15d, %r15d
	vmaskmovps	(%rbx,%r15,4), %ymm0, %ymm14
	vmaskmovps	(%r14,%r15,4), %ymm0, %ymm15
	vpsrld	$24, %ymm14, %ymm0
	vpmulld	%ymm7, %ymm0, %ymm0
	vpsrld	$8, %ymm0, %ymm0
	vpsrld	$24, %ymm15, %ymm2
	vpmulld	%ymm11, %ymm2, %ymm2
	vpsrld	$8, %ymm2, %ymm7
	vpand	%ymm3, %ymm2, %ymm12
	vpsubd	%ymm7, %ymm4, %ymm2
	vpmulld	%ymm0, %ymm2, %ymm2
	vpaddd	%ymm5, %ymm12, %ymm0
	vpaddd	%ymm0, %ymm2, %ymm0
	vpsrad	$8, %ymm0, %ymm7
	vpslld	$24, %ymm7, %ymm11
	vpminud	%ymm4, %ymm0, %ymm13
	vpcmpeqd	%ymm0, %ymm13, %ymm0
	vblendvps	%ymm0, %ymm8, %ymm7, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vdivps	%ymm0, %ymm9, %ymm13
	vpand	%ymm4, %ymm14, %ymm0
	vpand	%ymm4, %ymm15, %ymm7
	vpmulld	%ymm7, %ymm12, %ymm7
	vpmulld	%ymm0, %ymm2, %ymm0
	vpaddd	%ymm7, %ymm0, %ymm0
	vpshufb	%ymm10, %ymm15, %ymm7
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm13, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm5, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpor	%ymm0, %ymm11, %ymm0
	vpmulld	%ymm7, %ymm12, %ymm7
	vpshufb	%ymm10, %ymm14, %ymm11
	vpmulld	%ymm11, %ymm2, %ymm11
	vpaddd	%ymm7, %ymm11, %ymm7
	vpshufb	.LCPI1_6(%rip), %ymm15, %ymm11  # ymm11 = ymm15[2],zero,zero,zero,ymm15[6],zero,zero,zero,ymm15[10],zero,zero,zero,ymm15[14],zero,zero,zero,ymm15[18],zero,zero,zero,ymm15[22],zero,zero,zero,ymm15[26],zero,zero,zero,ymm15[30],zero,zero,zero
	vpmulld	%ymm11, %ymm12, %ymm11
	vpshufb	.LCPI1_6(%rip), %ymm14, %ymm12  # ymm12 = ymm14[2],zero,zero,zero,ymm14[6],zero,zero,zero,ymm14[10],zero,zero,zero,ymm14[14],zero,zero,zero,ymm14[18],zero,zero,zero,ymm14[22],zero,zero,zero,ymm14[26],zero,zero,zero,ymm14[30],zero,zero,zero
	vpmulld	%ymm12, %ymm2, %ymm2
	vpaddd	%ymm2, %ymm11, %ymm2
	vmovdqa	%ymm6, %ymm11
	vcvtdq2ps	%ymm7, %ymm7
	vmulps	%ymm7, %ymm13, %ymm7
	vcvtdq2ps	%ymm2, %ymm2
	vmulps	%ymm2, %ymm13, %ymm2
	vcvttps2dq	%ymm7, %ymm7
	vpaddd	%ymm5, %ymm7, %ymm7
	vpand	%ymm3, %ymm7, %ymm7
	vpor	%ymm7, %ymm0, %ymm0
	vmovdqu	-104(%rsp), %ymm7               # 32-byte Reload
	vcvttps2dq	%ymm2, %ymm2
	vpslld	$8, %ymm2, %ymm2
	vpaddd	%ymm1, %ymm2, %ymm2
	vpxor	%xmm12, %xmm12, %xmm12
	vpblendw	$170, %ymm2, %ymm12, %ymm2      # ymm2 = ymm12[0],ymm2[1],ymm12[2],ymm2[3],ymm12[4],ymm2[5],ymm12[6],ymm2[7],ymm12[8],ymm2[9],ymm12[10],ymm2[11],ymm12[12],ymm2[13],ymm12[14],ymm2[15]
	vpor	%ymm2, %ymm0, %ymm0
	vmovups	-40(%rsp), %ymm2                # 32-byte Reload
	vmaskmovps	%ymm0, %ymm2, (%rbx,%r15,4)
	jmp	.LBB1_7
.LBB1_8:
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
.LBB1_9:
	vzeroupper
	retq
.Lfunc_end1:
	.size	blit_rgba8888, .Lfunc_end1-blit_rgba8888
                                        # -- End function
	.ident	"Intel(r) Implicit SPMD Program Compiler (Intel(r) ISPC), 1.27.0 (build commit ae3af196790a1328 @ 20250515, LLVM 20.1.4)"
	.ident	"LLVM version 20.1.4 (https://github.com/llvm/llvm-project.git ec28b8f9cc7f2ac187d8a617a6d08d5e56f9120e)"
	.section	".note.GNU-stack","",@progbits

	.file	"blit.ispc"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function blit_rgba8888___un_3C_unu_3E_un_3C_Cunu_3E_uniuniuniuniuniuni
.LCPI0_0:
	.long	128                             # 0x80
.LCPI0_2:
	.long	256                             # 0x100
.LCPI0_3:
	.long	255                             # 0xff
.LCPI0_4:
	.long	1                               # 0x1
.LCPI0_5:
	.long	0x3f800000                      # float 1
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0
.LCPI0_1:
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
.LCPI0_6:
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
	pushq	%rax
                                        # kill: def $r8d killed $r8d def $r8
	movl	56(%rsp), %r10d
	movl	48(%rsp), %r11d
	vpcmpeqd	%ymm1, %ymm1, %ymm1
	cmpl	$255, %r11d
	movl	$256, %eax                      # imm = 0x100
	cmovel	%eax, %r11d
	cmpl	$255, %r10d
	cmovel	%eax, %r10d
	vtestps	%ymm1, %ymm0
	jb	.LBB0_14
# %bb.1:
	testl	%r9d, %r9d
	jle	.LBB0_27
# %bb.2:
	leal	7(%r8), %eax
	testl	%r8d, %r8d
	cmovnsl	%r8d, %eax
	andl	$-8, %eax
	vmovd	%r11d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vmovdqu	%ymm0, -64(%rsp)                # 32-byte Spill
	vmovd	%r10d, %xmm1
	vpbroadcastd	%xmm1, %ymm0
	vmovdqu	%ymm0, -96(%rsp)                # 32-byte Spill
	vmovd	%r8d, %xmm2
	vpbroadcastd	%xmm2, %ymm0
	vmovdqu	%ymm0, -128(%rsp)               # 32-byte Spill
	movslq	%eax, %r10
	xorl	%r11d, %r11d
	vpbroadcastd	.LCPI0_0(%rip), %ymm3   # ymm3 = [128,128,128,128,128,128,128,128]
	vpbroadcastd	.LCPI0_2(%rip), %ymm5   # ymm5 = [256,256,256,256,256,256,256,256]
	vpbroadcastd	.LCPI0_3(%rip), %ymm6   # ymm6 = [255,255,255,255,255,255,255,255]
	vbroadcastss	.LCPI0_4(%rip), %ymm7   # ymm7 = [1,1,1,1,1,1,1,1]
	vpcmpeqd	%ymm9, %ymm9, %ymm9
	jmp	.LBB0_3
	.p2align	4
.LBB0_12:                               #   in Loop: Header=BB0_3 Depth=1
	leaq	(%rbx,%r15,4), %rbx
	vmovdqu	-128(%rsp), %ymm0               # 32-byte Reload
	vpcmpgtd	-32(%rsp), %ymm0, %ymm0         # 32-byte Folded Reload
	vmaskmovps	%ymm10, %ymm0, (%rbx)
.LBB0_13:                               #   in Loop: Header=BB0_3 Depth=1
	incl	%r11d
	cmpl	%r9d, %r11d
	je	.LBB0_27
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
                                        #       Child Loop BB0_6 Depth 3
                                        #     Child Loop BB0_11 Depth 2
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
	jle	.LBB0_8
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=1
	xorl	%r15d, %r15d
	.p2align	4
.LBB0_5:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_6 Depth 3
	vmovdqu	(%rbx,%r15,4), %ymm11
	vmovdqu	(%r14,%r15,4), %ymm12
	vpsrld	$24, %ymm11, %ymm2
	vpmulld	-64(%rsp), %ymm2, %ymm2         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm2, %ymm2
	vpsrad	$8, %ymm2, %ymm2
	vpsrld	$24, %ymm12, %ymm4
	vpmulld	-96(%rsp), %ymm4, %ymm4         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm4, %ymm4
	vpsrad	$8, %ymm4, %ymm10
	vpsubd	%ymm10, %ymm5, %ymm13
	vpmulld	%ymm2, %ymm13, %ymm2
	vpand	.LCPI0_1(%rip), %ymm4, %ymm4
	vpaddd	%ymm3, %ymm4, %ymm4
	vpaddd	%ymm4, %ymm2, %ymm4
	vpsrad	$8, %ymm4, %ymm13
	vpminud	%ymm6, %ymm4, %ymm14
	vpcmpeqd	%ymm4, %ymm14, %ymm4
	vblendvps	%ymm4, %ymm7, %ymm13, %ymm4
	vcvtdq2ps	%ymm4, %ymm4
	vbroadcastss	.LCPI0_5(%rip), %ymm0   # ymm0 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vdivps	%ymm4, %ymm0, %ymm14
	vpminsd	%ymm6, %ymm13, %ymm4
	vpslld	$24, %ymm4, %ymm13
	vpxor	%xmm4, %xmm4, %xmm4
	vpcmpeqd	%ymm15, %ymm15, %ymm15
	.p2align	4
.LBB0_6:                                #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vpslld	$3, %ymm4, %ymm8
	vpsrlvd	%ymm8, %ymm11, %ymm0
	vpand	%ymm6, %ymm0, %ymm0
	vpsrlvd	%ymm8, %ymm12, %ymm1
	vpand	%ymm6, %ymm1, %ymm1
	vpmulld	%ymm10, %ymm1, %ymm1
	vpslld	$8, %ymm1, %ymm1
	vpmulld	%ymm2, %ymm0, %ymm0
	vpaddd	%ymm0, %ymm1, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm14, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm3, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpminsd	%ymm6, %ymm0, %ymm0
	vpsllvd	%ymm8, %ymm0, %ymm0
	vpor	%ymm0, %ymm13, %ymm0
	vblendvps	%ymm15, %ymm0, %ymm13, %ymm13
	vpminud	%ymm7, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vpsubd	%ymm9, %ymm4, %ymm4
	vtestps	%ymm15, %ymm0
	vpand	%ymm0, %ymm15, %ymm15
	jne	.LBB0_6
# %bb.7:                                #   in Loop: Header=BB0_5 Depth=2
	vmovups	%ymm13, (%rbx,%r15,4)
	addq	$8, %r15
	cmpq	%r10, %r15
	jl	.LBB0_5
.LBB0_8:                                #   in Loop: Header=BB0_3 Depth=1
	cmpl	%r8d, %r15d
	jge	.LBB0_13
# %bb.9:                                #   in Loop: Header=BB0_3 Depth=1
	vmovd	%r15d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vpor	.LCPI0_6(%rip), %ymm0, %ymm1
	movl	%r15d, %r15d
	vmovdqu	-128(%rsp), %ymm0               # 32-byte Reload
	vmovdqu	%ymm1, -32(%rsp)                # 32-byte Spill
	vpcmpgtd	%ymm1, %ymm0, %ymm13
	vmaskmovps	(%rbx,%r15,4), %ymm13, %ymm12
	vmaskmovps	(%r14,%r15,4), %ymm13, %ymm14
	vpsrld	$24, %ymm12, %ymm0
	vpmulld	-64(%rsp), %ymm0, %ymm0         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpsrld	$24, %ymm14, %ymm1
	vpmulld	-96(%rsp), %ymm1, %ymm1         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm1, %ymm1
	vpsrad	$8, %ymm1, %ymm15
	vpand	.LCPI0_1(%rip), %ymm1, %ymm1
	vpsubd	%ymm15, %ymm5, %ymm2
	vpmulld	%ymm0, %ymm2, %ymm2
	vpaddd	%ymm3, %ymm1, %ymm0
	vpaddd	%ymm0, %ymm2, %ymm4
	vpsrad	$8, %ymm4, %ymm11
	vpminsd	%ymm6, %ymm11, %ymm0
	vpslld	$24, %ymm0, %ymm10
	vtestps	%ymm13, %ymm13
	je	.LBB0_12
# %bb.10:                               #   in Loop: Header=BB0_3 Depth=1
	vpminud	%ymm6, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vblendvps	%ymm0, %ymm7, %ymm11, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vbroadcastss	.LCPI0_5(%rip), %ymm1   # ymm1 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vdivps	%ymm0, %ymm1, %ymm11
	vpxor	%xmm4, %xmm4, %xmm4
	.p2align	4
.LBB0_11:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vpslld	$3, %ymm4, %ymm0
	vpsrlvd	%ymm0, %ymm12, %ymm1
	vpand	%ymm6, %ymm1, %ymm1
	vpsrlvd	%ymm0, %ymm14, %ymm8
	vpand	%ymm6, %ymm8, %ymm8
	vpmulld	%ymm15, %ymm8, %ymm8
	vpslld	$8, %ymm8, %ymm8
	vpmulld	%ymm2, %ymm1, %ymm1
	vpaddd	%ymm1, %ymm8, %ymm1
	vcvtdq2ps	%ymm1, %ymm1
	vmulps	%ymm1, %ymm11, %ymm1
	vcvttps2dq	%ymm1, %ymm1
	vpaddd	%ymm3, %ymm1, %ymm1
	vpsrad	$8, %ymm1, %ymm1
	vpminsd	%ymm6, %ymm1, %ymm1
	vpsllvd	%ymm0, %ymm1, %ymm0
	vpor	%ymm0, %ymm10, %ymm0
	vblendvps	%ymm13, %ymm0, %ymm10, %ymm10
	vpminud	%ymm7, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vpsubd	%ymm9, %ymm4, %ymm4
	vtestps	%ymm13, %ymm0
	vandps	%ymm0, %ymm13, %ymm13
	jne	.LBB0_11
	jmp	.LBB0_12
.LBB0_14:
	testl	%r9d, %r9d
	jle	.LBB0_27
# %bb.15:
	leal	7(%r8), %eax
	testl	%r8d, %r8d
	cmovnsl	%r8d, %eax
	andl	$-8, %eax
	vmovd	%r11d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vmovdqu	%ymm0, -64(%rsp)                # 32-byte Spill
	vmovd	%r10d, %xmm1
	vpbroadcastd	%xmm1, %ymm0
	vmovdqu	%ymm0, -96(%rsp)                # 32-byte Spill
	vmovd	%r8d, %xmm2
	vpbroadcastd	%xmm2, %ymm0
	vmovdqu	%ymm0, -128(%rsp)               # 32-byte Spill
	movslq	%eax, %r10
	xorl	%r11d, %r11d
	vpbroadcastd	.LCPI0_0(%rip), %ymm3   # ymm3 = [128,128,128,128,128,128,128,128]
	vpbroadcastd	.LCPI0_2(%rip), %ymm5   # ymm5 = [256,256,256,256,256,256,256,256]
	vpbroadcastd	.LCPI0_3(%rip), %ymm6   # ymm6 = [255,255,255,255,255,255,255,255]
	vbroadcastss	.LCPI0_4(%rip), %ymm7   # ymm7 = [1,1,1,1,1,1,1,1]
	vpcmpeqd	%ymm9, %ymm9, %ymm9
	jmp	.LBB0_16
	.p2align	4
.LBB0_25:                               #   in Loop: Header=BB0_16 Depth=1
	leaq	(%rbx,%r15,4), %rbx
	vmovdqu	-128(%rsp), %ymm0               # 32-byte Reload
	vpcmpgtd	-32(%rsp), %ymm0, %ymm0         # 32-byte Folded Reload
	vmaskmovps	%ymm10, %ymm0, (%rbx)
.LBB0_26:                               #   in Loop: Header=BB0_16 Depth=1
	incl	%r11d
	cmpl	%r9d, %r11d
	je	.LBB0_27
.LBB0_16:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_18 Depth 2
                                        #       Child Loop BB0_19 Depth 3
                                        #     Child Loop BB0_24 Depth 2
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
	jle	.LBB0_21
# %bb.17:                               #   in Loop: Header=BB0_16 Depth=1
	xorl	%r15d, %r15d
	.p2align	4
.LBB0_18:                               #   Parent Loop BB0_16 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_19 Depth 3
	vmovdqu	(%rbx,%r15,4), %ymm11
	vmovdqu	(%r14,%r15,4), %ymm12
	vpsrld	$24, %ymm11, %ymm2
	vpmulld	-64(%rsp), %ymm2, %ymm2         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm2, %ymm2
	vpsrad	$8, %ymm2, %ymm2
	vpsrld	$24, %ymm12, %ymm4
	vpmulld	-96(%rsp), %ymm4, %ymm4         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm4, %ymm4
	vpsrad	$8, %ymm4, %ymm10
	vpsubd	%ymm10, %ymm5, %ymm13
	vpmulld	%ymm2, %ymm13, %ymm2
	vpand	.LCPI0_1(%rip), %ymm4, %ymm4
	vpaddd	%ymm3, %ymm4, %ymm4
	vpaddd	%ymm4, %ymm2, %ymm4
	vpsrad	$8, %ymm4, %ymm13
	vpminud	%ymm6, %ymm4, %ymm14
	vpcmpeqd	%ymm4, %ymm14, %ymm4
	vblendvps	%ymm4, %ymm7, %ymm13, %ymm4
	vcvtdq2ps	%ymm4, %ymm4
	vbroadcastss	.LCPI0_5(%rip), %ymm0   # ymm0 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vdivps	%ymm4, %ymm0, %ymm14
	vpminsd	%ymm6, %ymm13, %ymm4
	vpslld	$24, %ymm4, %ymm13
	vpxor	%xmm4, %xmm4, %xmm4
	vpcmpeqd	%ymm15, %ymm15, %ymm15
	.p2align	4
.LBB0_19:                               #   Parent Loop BB0_16 Depth=1
                                        #     Parent Loop BB0_18 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vpslld	$3, %ymm4, %ymm8
	vpsrlvd	%ymm8, %ymm11, %ymm0
	vpand	%ymm6, %ymm0, %ymm0
	vpsrlvd	%ymm8, %ymm12, %ymm1
	vpand	%ymm6, %ymm1, %ymm1
	vpmulld	%ymm10, %ymm1, %ymm1
	vpslld	$8, %ymm1, %ymm1
	vpmulld	%ymm2, %ymm0, %ymm0
	vpaddd	%ymm0, %ymm1, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm14, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm3, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpminsd	%ymm6, %ymm0, %ymm0
	vpsllvd	%ymm8, %ymm0, %ymm0
	vpor	%ymm0, %ymm13, %ymm0
	vblendvps	%ymm15, %ymm0, %ymm13, %ymm13
	vpminud	%ymm7, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vpsubd	%ymm9, %ymm4, %ymm4
	vtestps	%ymm15, %ymm0
	vpand	%ymm0, %ymm15, %ymm15
	jne	.LBB0_19
# %bb.20:                               #   in Loop: Header=BB0_18 Depth=2
	vmovups	%ymm13, (%rbx,%r15,4)
	addq	$8, %r15
	cmpq	%r10, %r15
	jl	.LBB0_18
.LBB0_21:                               #   in Loop: Header=BB0_16 Depth=1
	cmpl	%r8d, %r15d
	jge	.LBB0_26
# %bb.22:                               #   in Loop: Header=BB0_16 Depth=1
	vmovd	%r15d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vpor	.LCPI0_6(%rip), %ymm0, %ymm1
	movl	%r15d, %r15d
	vmovdqu	-128(%rsp), %ymm0               # 32-byte Reload
	vmovdqu	%ymm1, -32(%rsp)                # 32-byte Spill
	vpcmpgtd	%ymm1, %ymm0, %ymm13
	vmaskmovps	(%rbx,%r15,4), %ymm13, %ymm12
	vmaskmovps	(%r14,%r15,4), %ymm13, %ymm14
	vpsrld	$24, %ymm12, %ymm0
	vpmulld	-64(%rsp), %ymm0, %ymm0         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpsrld	$24, %ymm14, %ymm1
	vpmulld	-96(%rsp), %ymm1, %ymm1         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm1, %ymm1
	vpsrad	$8, %ymm1, %ymm15
	vpand	.LCPI0_1(%rip), %ymm1, %ymm1
	vpsubd	%ymm15, %ymm5, %ymm2
	vpmulld	%ymm0, %ymm2, %ymm2
	vpaddd	%ymm3, %ymm1, %ymm0
	vpaddd	%ymm0, %ymm2, %ymm4
	vpsrad	$8, %ymm4, %ymm11
	vpminsd	%ymm6, %ymm11, %ymm0
	vpslld	$24, %ymm0, %ymm10
	vtestps	%ymm13, %ymm13
	je	.LBB0_25
# %bb.23:                               #   in Loop: Header=BB0_16 Depth=1
	vpminud	%ymm6, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vblendvps	%ymm0, %ymm7, %ymm11, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vbroadcastss	.LCPI0_5(%rip), %ymm1   # ymm1 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vdivps	%ymm0, %ymm1, %ymm11
	vpxor	%xmm4, %xmm4, %xmm4
	.p2align	4
.LBB0_24:                               #   Parent Loop BB0_16 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vpslld	$3, %ymm4, %ymm0
	vpsrlvd	%ymm0, %ymm12, %ymm1
	vpand	%ymm6, %ymm1, %ymm1
	vpsrlvd	%ymm0, %ymm14, %ymm8
	vpand	%ymm6, %ymm8, %ymm8
	vpmulld	%ymm15, %ymm8, %ymm8
	vpslld	$8, %ymm8, %ymm8
	vpmulld	%ymm2, %ymm1, %ymm1
	vpaddd	%ymm1, %ymm8, %ymm1
	vcvtdq2ps	%ymm1, %ymm1
	vmulps	%ymm1, %ymm11, %ymm1
	vcvttps2dq	%ymm1, %ymm1
	vpaddd	%ymm3, %ymm1, %ymm1
	vpsrad	$8, %ymm1, %ymm1
	vpminsd	%ymm6, %ymm1, %ymm1
	vpsllvd	%ymm0, %ymm1, %ymm0
	vpor	%ymm0, %ymm10, %ymm0
	vblendvps	%ymm13, %ymm0, %ymm10, %ymm10
	vpminud	%ymm7, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vpsubd	%ymm9, %ymm4, %ymm4
	vtestps	%ymm13, %ymm0
	vandps	%ymm0, %ymm13, %ymm13
	jne	.LBB0_24
	jmp	.LBB0_25
.LBB0_27:
	addq	$8, %rsp
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
	.long	128                             # 0x80
.LCPI1_2:
	.long	256                             # 0x100
.LCPI1_3:
	.long	255                             # 0xff
.LCPI1_4:
	.long	1                               # 0x1
.LCPI1_5:
	.long	0x3f800000                      # float 1
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0
.LCPI1_1:
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	0                               # 0x0
	.byte	255                             # 0xff
	.byte	255                             # 0xff
	.byte	255                             # 0xff
.LCPI1_6:
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
	jle	.LBB1_14
# %bb.1:
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	pushq	%rax
	movl	56(%rsp), %r10d
	movl	48(%rsp), %r11d
	cmpl	$255, %r10d
	movl	$256, %eax                      # imm = 0x100
	cmovel	%eax, %r10d
	cmpl	$255, %r11d
	cmovel	%eax, %r11d
	leal	7(%r8), %eax
	testl	%r8d, %r8d
	cmovnsl	%r8d, %eax
	andl	$-8, %eax
	vmovd	%r11d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vmovdqu	%ymm0, -64(%rsp)                # 32-byte Spill
	vmovd	%r10d, %xmm1
	vpbroadcastd	%xmm1, %ymm0
	vmovdqu	%ymm0, -96(%rsp)                # 32-byte Spill
	vmovd	%r8d, %xmm2
	vpbroadcastd	%xmm2, %ymm0
	vmovdqu	%ymm0, -128(%rsp)               # 32-byte Spill
	movslq	%eax, %r10
	xorl	%r11d, %r11d
	vpbroadcastd	.LCPI1_0(%rip), %ymm3   # ymm3 = [128,128,128,128,128,128,128,128]
	vpbroadcastd	.LCPI1_2(%rip), %ymm5   # ymm5 = [256,256,256,256,256,256,256,256]
	vpbroadcastd	.LCPI1_3(%rip), %ymm6   # ymm6 = [255,255,255,255,255,255,255,255]
	vbroadcastss	.LCPI1_4(%rip), %ymm7   # ymm7 = [1,1,1,1,1,1,1,1]
	vpcmpeqd	%ymm9, %ymm9, %ymm9
	jmp	.LBB1_2
	.p2align	4
.LBB1_11:                               #   in Loop: Header=BB1_2 Depth=1
	leaq	(%rbx,%r15,4), %rbx
	vmovdqu	-128(%rsp), %ymm0               # 32-byte Reload
	vpcmpgtd	-32(%rsp), %ymm0, %ymm0         # 32-byte Folded Reload
	vmaskmovps	%ymm10, %ymm0, (%rbx)
.LBB1_12:                               #   in Loop: Header=BB1_2 Depth=1
	incl	%r11d
	cmpl	%r9d, %r11d
	je	.LBB1_13
.LBB1_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_4 Depth 2
                                        #       Child Loop BB1_5 Depth 3
                                        #     Child Loop BB1_10 Depth 2
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
	jle	.LBB1_7
# %bb.3:                                #   in Loop: Header=BB1_2 Depth=1
	xorl	%r15d, %r15d
	.p2align	4
.LBB1_4:                                #   Parent Loop BB1_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_5 Depth 3
	vmovdqu	(%rbx,%r15,4), %ymm11
	vmovdqu	(%r14,%r15,4), %ymm12
	vpsrld	$24, %ymm11, %ymm2
	vpmulld	-64(%rsp), %ymm2, %ymm2         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm2, %ymm2
	vpsrad	$8, %ymm2, %ymm2
	vpsrld	$24, %ymm12, %ymm4
	vpmulld	-96(%rsp), %ymm4, %ymm4         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm4, %ymm4
	vpsrad	$8, %ymm4, %ymm10
	vpsubd	%ymm10, %ymm5, %ymm13
	vpmulld	%ymm2, %ymm13, %ymm2
	vpand	.LCPI1_1(%rip), %ymm4, %ymm4
	vpaddd	%ymm3, %ymm4, %ymm4
	vpaddd	%ymm4, %ymm2, %ymm4
	vpsrad	$8, %ymm4, %ymm13
	vpminud	%ymm6, %ymm4, %ymm14
	vpcmpeqd	%ymm4, %ymm14, %ymm4
	vblendvps	%ymm4, %ymm7, %ymm13, %ymm4
	vcvtdq2ps	%ymm4, %ymm4
	vbroadcastss	.LCPI1_5(%rip), %ymm0   # ymm0 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vdivps	%ymm4, %ymm0, %ymm14
	vpminsd	%ymm6, %ymm13, %ymm4
	vpslld	$24, %ymm4, %ymm13
	vpxor	%xmm4, %xmm4, %xmm4
	vpcmpeqd	%ymm15, %ymm15, %ymm15
	.p2align	4
.LBB1_5:                                #   Parent Loop BB1_2 Depth=1
                                        #     Parent Loop BB1_4 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vpslld	$3, %ymm4, %ymm8
	vpsrlvd	%ymm8, %ymm11, %ymm0
	vpand	%ymm6, %ymm0, %ymm0
	vpsrlvd	%ymm8, %ymm12, %ymm1
	vpand	%ymm6, %ymm1, %ymm1
	vpmulld	%ymm10, %ymm1, %ymm1
	vpslld	$8, %ymm1, %ymm1
	vpmulld	%ymm2, %ymm0, %ymm0
	vpaddd	%ymm0, %ymm1, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vmulps	%ymm0, %ymm14, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm3, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpminsd	%ymm6, %ymm0, %ymm0
	vpsllvd	%ymm8, %ymm0, %ymm0
	vpor	%ymm0, %ymm13, %ymm0
	vblendvps	%ymm15, %ymm0, %ymm13, %ymm13
	vpminud	%ymm7, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vpsubd	%ymm9, %ymm4, %ymm4
	vtestps	%ymm15, %ymm0
	vpand	%ymm0, %ymm15, %ymm15
	jne	.LBB1_5
# %bb.6:                                #   in Loop: Header=BB1_4 Depth=2
	vmovups	%ymm13, (%rbx,%r15,4)
	addq	$8, %r15
	cmpq	%r10, %r15
	jl	.LBB1_4
.LBB1_7:                                #   in Loop: Header=BB1_2 Depth=1
	cmpl	%r8d, %r15d
	jge	.LBB1_12
# %bb.8:                                #   in Loop: Header=BB1_2 Depth=1
	vmovd	%r15d, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	vpor	.LCPI1_6(%rip), %ymm0, %ymm1
	movl	%r15d, %r15d
	vmovdqu	-128(%rsp), %ymm0               # 32-byte Reload
	vmovdqu	%ymm1, -32(%rsp)                # 32-byte Spill
	vpcmpgtd	%ymm1, %ymm0, %ymm13
	vmaskmovps	(%rbx,%r15,4), %ymm13, %ymm12
	vmaskmovps	(%r14,%r15,4), %ymm13, %ymm14
	vpsrld	$24, %ymm12, %ymm0
	vpmulld	-64(%rsp), %ymm0, %ymm0         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm0, %ymm0
	vpsrad	$8, %ymm0, %ymm0
	vpsrld	$24, %ymm14, %ymm1
	vpmulld	-96(%rsp), %ymm1, %ymm1         # 32-byte Folded Reload
	vpaddd	%ymm3, %ymm1, %ymm1
	vpsrad	$8, %ymm1, %ymm15
	vpand	.LCPI1_1(%rip), %ymm1, %ymm1
	vpsubd	%ymm15, %ymm5, %ymm2
	vpmulld	%ymm0, %ymm2, %ymm2
	vpaddd	%ymm3, %ymm1, %ymm0
	vpaddd	%ymm0, %ymm2, %ymm4
	vpsrad	$8, %ymm4, %ymm11
	vpminsd	%ymm6, %ymm11, %ymm0
	vpslld	$24, %ymm0, %ymm10
	vtestps	%ymm13, %ymm13
	je	.LBB1_11
# %bb.9:                                #   in Loop: Header=BB1_2 Depth=1
	vpminud	%ymm6, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vblendvps	%ymm0, %ymm7, %ymm11, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vbroadcastss	.LCPI1_5(%rip), %ymm1   # ymm1 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	vdivps	%ymm0, %ymm1, %ymm11
	vpxor	%xmm4, %xmm4, %xmm4
	.p2align	4
.LBB1_10:                               #   Parent Loop BB1_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vpslld	$3, %ymm4, %ymm0
	vpsrlvd	%ymm0, %ymm12, %ymm1
	vpand	%ymm6, %ymm1, %ymm1
	vpsrlvd	%ymm0, %ymm14, %ymm8
	vpand	%ymm6, %ymm8, %ymm8
	vpmulld	%ymm15, %ymm8, %ymm8
	vpslld	$8, %ymm8, %ymm8
	vpmulld	%ymm2, %ymm1, %ymm1
	vpaddd	%ymm1, %ymm8, %ymm1
	vcvtdq2ps	%ymm1, %ymm1
	vmulps	%ymm1, %ymm11, %ymm1
	vcvttps2dq	%ymm1, %ymm1
	vpaddd	%ymm3, %ymm1, %ymm1
	vpsrad	$8, %ymm1, %ymm1
	vpminsd	%ymm6, %ymm1, %ymm1
	vpsllvd	%ymm0, %ymm1, %ymm0
	vpor	%ymm0, %ymm10, %ymm0
	vblendvps	%ymm13, %ymm0, %ymm10, %ymm10
	vpminud	%ymm7, %ymm4, %ymm0
	vpcmpeqd	%ymm0, %ymm4, %ymm0
	vpsubd	%ymm9, %ymm4, %ymm4
	vtestps	%ymm13, %ymm0
	vandps	%ymm0, %ymm13, %ymm13
	jne	.LBB1_10
	jmp	.LBB1_11
.LBB1_13:
	addq	$8, %rsp
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
.LBB1_14:
	vzeroupper
	retq
.Lfunc_end1:
	.size	blit_rgba8888, .Lfunc_end1-blit_rgba8888
                                        # -- End function
	.ident	"Intel(r) Implicit SPMD Program Compiler (Intel(r) ISPC), 1.27.0 (build commit ae3af196790a1328 @ 20250515, LLVM 20.1.4)"
	.ident	"LLVM version 20.1.4 (https://github.com/llvm/llvm-project.git ec28b8f9cc7f2ac187d8a617a6d08d5e56f9120e)"
	.section	".note.GNU-stack","",@progbits
